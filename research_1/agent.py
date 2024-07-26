import os
from typing import List, Optional, Any
import typer
from transformers import AutoTokenizer, AutoModel
import torch
from sqlmodel import Field, Session, SQLModel, create_engine, select
from pgvector.sqlalchemy import Vector
from pydantic import BaseModel
import instructor
from openai import OpenAI
from anthropic import Anthropic
from abc import ABC, abstractmethod
import sys
import subprocess
import queue
import pydantic
import time
from scipy.spatial.distance import cosine
from sqlalchemy import text
from datetime import datetime

# Initialize the OpenAI and Anthropic clients
openai_client = OpenAI()
anthropic_client = Anthropic()


class AbstractMemory(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    content: str
    embedding: List[float] = Field(sa_column=Vector(384))
    do_not_delete: bool = Field(default=False)

    __table_args__ = {"extend_existing": True}


class GroundTruthMemory(AbstractMemory, table=True):
    __tablename__ = "ground_truth_memory"


class RelationMemory(AbstractMemory, table=True):
    __tablename__ = "relation_memory"
    src_id: int = Field(foreign_key="abstract_memory.id")
    dst_id: int = Field(foreign_key="abstract_memory.id")


class SyntheticRelationMemory(RelationMemory, table=True):
    __tablename__ = "synthetic_relation_memory"
    confidence: float


class TemporalRelationMemory(RelationMemory, table=True):
    __tablename__ = "temporal_relation_memory"
    timestamp: datetime


class ExtractedMemories(BaseModel):
    memories: List[str] = Field(
        description="List of extracted memories relevant to the input"
    )


def parse_model(model: type[BaseModel], prompt: str, **kwargs) -> BaseModel:
    """Parse the response using the Instructor library."""
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=model,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )


class Agent(BaseModel, ABC):
    @abstractmethod
    def input(self, data: Any, **kwargs):
        pass

    @abstractmethod
    def output(self, **kwargs) -> Any:
        pass


class RAGAgent(Agent):
    OPENAI_MODEL = "gpt-4-turbo-preview"
    ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
    SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_TOKENS = 300
    INITIAL_KNOWLEDGE = [
        "I am an experimental AI system",
        "I am continuously improving through feedback and new interactions",
    ]
    DB_URL = "postgresql://user:password@localhost/ragagent"

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.SENTENCE_TRANSFORMER_MODEL)
        self.embedding_model = AutoModel.from_pretrained(
            self.SENTENCE_TRANSFORMER_MODEL
        )

        self.engine = create_engine(self.DB_URL)
        SQLModel.metadata.create_all(self.engine)

        self.create_stored_procedures()
        self.load_state()

    def create_stored_procedures(self):
        with self.engine.connect() as connection:
            connection.execute(
                text(
                    """
                -- Stored procedure for calculate_hop_distance
                CREATE OR REPLACE FUNCTION calculate_hop_distance(start_id INTEGER, end_id INTEGER)
                RETURNS INTEGER AS $$
                DECLARE
                    visited INTEGER[];
                    queue INTEGER[][];
                    current_id INTEGER;
                    distance INTEGER;
                    relation RECORD;
                BEGIN
                    visited := ARRAY[]::INTEGER[];
                    queue := ARRAY[[start_id, 0]]::INTEGER[][];

                    WHILE array_length(queue, 1) > 0 LOOP
                        current_id := queue[1][1];
                        distance := queue[1][2];
                        queue := queue[2:];

                        IF current_id = end_id THEN
                            RETURN distance;
                        END IF;

                        IF NOT (current_id = ANY(visited)) THEN
                            visited := array_append(visited, current_id);
                            FOR relation IN SELECT dst_id FROM memory_relation WHERE src_id = current_id LOOP
                                IF NOT (relation.dst_id = ANY(visited)) THEN
                                    queue := array_cat(queue, ARRAY[[relation.dst_id, distance + 1]]::INTEGER[][]);
                                END IF;
                            END LOOP;
                        END IF;
                    END LOOP;

                    RETURN -1; -- No path found
                END;
                $$ LANGUAGE plpgsql;

                -- Function for cosine distance calculation
                CREATE OR REPLACE FUNCTION cosine_distance(vector1 FLOAT[], vector2 FLOAT[])
                RETURNS FLOAT AS $$
                DECLARE
                    dot_product FLOAT := 0;
                    magnitude1 FLOAT := 0;
                    magnitude2 FLOAT := 0;
                BEGIN
                    FOR i IN 1..array_length(vector1, 1) LOOP
                        dot_product := dot_product + (vector1[i] * vector2[i]);
                        magnitude1 := magnitude1 + (vector1[i] * vector1[i]);
                        magnitude2 := magnitude2 + (vector2[i] * vector2[i]);
                    END LOOP;

                    magnitude1 := sqrt(magnitude1);
                    magnitude2 := sqrt(magnitude2);

                    IF magnitude1 = 0 OR magnitude2 = 0 THEN
                        RETURN 1.0;
                    ELSE
                        RETURN 1.0 - (dot_product / (magnitude1 * magnitude2));
                    END IF;
                END;
                $$ LANGUAGE plpgsql;
                """
                )
            )
            connection.commit()

    def embed_text(self, text: str) -> List[float]:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

    def load_state(self):
        with Session(self.engine) as session:
            if session.exec(select(AbstractMemory)).first() is None:
                for knowledge in self.INITIAL_KNOWLEDGE:
                    self.update_knowledge_base(knowledge)

    def update_knowledge_base(
        self, new_info: str, do_not_delete: bool = False, is_ground_truth: bool = False
    ):
        embedding = self.embed_text(new_info)
        if is_ground_truth:
            memory = GroundTruthMemory(
                content=new_info, embedding=embedding, do_not_delete=do_not_delete
            )
        else:
            memory = AbstractMemory(
                content=new_info, embedding=embedding, do_not_delete=do_not_delete
            )
        with Session(self.engine) as session:
            session.add(memory)
            session.commit()

    def add_memory_relation(
        self,
        src_id: int,
        dst_id: int,
        relation: str,
        confidence: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ):
        with Session(self.engine) as session:
            if confidence is not None:
                memory_relation = SyntheticRelationMemory(
                    content=relation,
                    src_id=src_id,
                    dst_id=dst_id,
                    confidence=confidence,
                )
            elif timestamp is not None:
                memory_relation = TemporalRelationMemory(
                    content=relation,
                    src_id=src_id,
                    dst_id=dst_id,
                    timestamp=timestamp,
                )
            else:
                memory_relation = RelationMemory(
                    content=relation, src_id=src_id, dst_id=dst_id
                )
            session.add(memory_relation)
            session.commit()

    def calculate_hop_distance(self, start_id: int, end_id: int) -> int:
        with self.engine.connect() as connection:
            result = connection.execute(
                text("SELECT calculate_hop_distance(:start_id, :end_id)"),
                {"start_id": start_id, "end_id": end_id},
            )
            return result.scalar_one()

    def identify_connections(
        self,
        k: int = 5,
        similarity_threshold: float = 0.5,
        hop_distance_threshold: int = 1,
    ):
        with self.engine.connect() as connection:
            result = connection.execute(
                text(
                    """
                    WITH memory_pairs AS (
                        SELECT 
                            m1.id AS memory1_id,
                            m1.content AS memory1_content,
                            m2.id AS memory2_id,
                            m2.content AS memory2_content,
                            1 - cosine_distance(m1.embedding, m2.embedding) AS similarity,
                            calculate_hop_distance(m1.id, m2.id) AS hop_distance
                        FROM abstract_memory m1
                        CROSS JOIN abstract_memory m2
                        WHERE m1.id != m2.id
                    )
                    SELECT 
                        memory1_content,
                        memory2_content,
                        similarity,
                        hop_distance,
                        similarity * (1.0 / NULLIF(hop_distance, 0)) AS connection_strength
                    FROM memory_pairs
                    WHERE 
                        similarity > :sim_threshold AND 
                        hop_distance > :hop_threshold
                    ORDER BY connection_strength DESC
                    LIMIT :k
                    """
                ),
                {
                    "sim_threshold": similarity_threshold,
                    "hop_threshold": hop_distance_threshold,
                    "k": k,
                },
            )

            connections = []
            for row in result:
                connection = self.describe_connection(
                    row.memory1_content, row.memory2_content
                )
                connections.append(
                    {
                        "connection": connection,
                        "memory1": row.memory1_content,
                        "memory2": row.memory2_content,
                        "similarity": row.similarity,
                        "hop_distance": row.hop_distance,
                        "connection_strength": row.connection_strength,
                    }
                )
            return connections

    def describe_connection(self, content1: str, content2: str) -> str:
        prompt = f"Given these two pieces of information:\n1. {content1}\n2. {content2}\n\nDescribe the connection or relationship between these two pieces of information."

        response = openai_client.chat.completions.create(
            model=self.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant tasked with identifying and describing connections between pieces of information.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=100,
        )

        return response.choices[0].message.content.strip()

    def calculate_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        return 1 - cosine(embedding1, embedding2)

    def remove_from_knowledge_base(self, info_to_remove: List[str]):
        with Session(self.engine) as session:
            for info in info_to_remove:
                memory = session.exec(
                    select(AbstractMemory).where(
                        AbstractMemory.content == info,
                        AbstractMemory.do_not_delete == False,
                    )
                ).first()
                if memory:
                    session.delete(memory)
            session.commit()

    def remove_related_memories(self, query: str, k: int = 5):
        embedding = self.embed_text(query)
        with Session(self.engine) as session:
            related_memories = session.exec(
                select(AbstractMemory)
                .where(AbstractMemory.do_not_delete == False)
                .order_by(AbstractMemory.embedding.cosine_distance(embedding))
                .limit(k)
            ).all()
            for memory in related_memories:
                session.delete(memory)
            session.commit()

    def set_memory_do_not_delete(self, memory_id: int, do_not_delete: bool = True):
        with Session(self.engine) as session:
            memory = session.get(AbstractMemory, memory_id)
            if memory:
                memory.do_not_delete = do_not_delete
                session.add(memory)
                session.commit()
                return True
            return False

    def get_all_memories(self) -> List[dict]:
        with Session(self.engine) as session:
            memories = session.exec(select(AbstractMemory)).all()
            return [
                {
                    "id": m.id,
                    "content": m.content,
                    "do_not_delete": m.do_not_delete,
                    "type": type(m).__name__,
                }
                for m in memories
            ]

    def get_memory_relations(self, memory_id: int) -> List[dict]:
        with Session(self.engine) as session:
            relations = session.exec(
                select(RelationMemory).where(
                    (RelationMemory.src_id == memory_id)
                    | (RelationMemory.dst_id == memory_id)
                )
            ).all()
            return [
                {
                    "src_id": r.src_id,
                    "dst_id": r.dst_id,
                    "relation_type": r.content,
                    "type": type(r).__name__,
                    "confidence": getattr(r, "confidence", None),
                    "timestamp": getattr(r, "timestamp", None),
                }
                for r in relations
            ]

    def input(self, data: Any, **kwargs):
        if isinstance(data, str):
            self.update_knowledge_base(data)
            self.last_query = data
        elif isinstance(data, list):
            for item in data:
                self.update_knowledge_base(item)
            self.last_query = data[-1] if data else ""
        else:
            raise ValueError("Input data must be a string or a list of strings")

        # Perception part (previously in generate_response)
        embedding = self.embed_text(self.last_query)
        with Session(self.engine) as session:
            related_memories = session.exec(
                select(AbstractMemory)
                .order_by(AbstractMemory.embedding.cosine_distance(embedding))
                .limit(5)
            ).all()
            self.context = [memory.content for memory in related_memories]

        # Extract relevant memories using GPT-4-turbo
        response = openai_client.chat.completions.create(
            model=self.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Extract a list of relevant memories from the given context and user input.",
                },
                {
                    "role": "user",
                    "content": f"Current memories: {self.context}\n\nUser input: {self.last_query}\n\nExtract a list of relevant memories:",
                },
            ],
            max_tokens=150,
        )
        self.extracted_memories = response.choices[0].message.content.split("\n")

    def output(self, **kwargs) -> Any:
        if not hasattr(self, "last_query") or not hasattr(self, "extracted_memories"):
            raise ValueError("Input must be called before output")

        # Generate response using Claude 3.5 Sonnet
        response = anthropic_client.messages.create(
            model=self.ANTHROPIC_MODEL,
            max_tokens=self.MAX_TOKENS,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {
                    "role": "user",
                    "content": f"Context: {' '.join(self.extracted_memories)}\n\nQuestion: {self.last_query}",
                },
            ],
        )
        return response.content[0].text

    def feed_file(self, file_path: str):
        with open(file_path, "r") as file:
            content = file.read()

        # Simple text splitting (you may want to implement a more sophisticated method)
        chunks = [content[i : i + 1000] for i in range(0, len(content), 1000)]

        novel_pieces = []
        for chunk in chunks:
            # Here you'd typically use an LLM to extract novel information
            # For simplicity, we'll just add the chunk if it's not already in the database
            if not self.is_information_exists(chunk):
                self.update_knowledge_base(chunk)
                novel_pieces.append(chunk)

        return f"Extracted and added {len(novel_pieces)} novel pieces of information from {file_path}"

    def is_information_exists(self, info: str) -> bool:
        with Session(self.engine) as session:
            exists = (
                session.exec(
                    select(AbstractMemory).where(AbstractMemory.content == info)
                ).first()
                is not None
            )
        return exists


class HumanAgent(Agent):
    def input(self, data: Any, **kwargs):
        self.last_query = data if isinstance(data, str) else str(data)
        print(f"\nQuestion: {self.last_query}")

    def output(self, **kwargs) -> Any:
        if not hasattr(self, "last_query"):
            raise ValueError("Input must be called before output")
        print("Please provide your response:")
        return sys.stdin.readline().strip()


class LineReaderBot(Agent):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.current_line = 0
        self.total_lines = sum(1 for _ in open(self.file_path, "r"))
        self.file = open(self.file_path, "r")
        self.current_line = next(self.file)

    def __del__(self):
        self.file.close()

    def input(self, data: Any, **kwargs):
        pass

    def output(self, **kwargs) -> Any:
        print(self.current_line)
        self.current_line = next(self.file)


class ProgramWrapperAgent(Agent):
    def __init__(self, command: List[str], timeout: float = 5.0):
        self.command = command
        self.timeout = timeout
        self.process = None
        self.output_queue = queue.Queue()

    def input(self, data: Any, **kwargs):
        self.last_input = data

    def output(self, **kwargs) -> Any:
        if not hasattr(self, "last_input"):
            raise ValueError("Input must be called before output")

        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        def enqueue_output(out, queue):
            for line in iter(out.readline, ""):
                queue.put(line)
            out.close()

        t = threading.Thread(
            target=enqueue_output, args=(self.process.stdout, self.output_queue)
        )
        t.daemon = True
        t.start()

        self.process.stdin.write(f"{self.last_input}\n")
        self.process.stdin.flush()

        output = []
        end_time = time.time() + self.timeout

        while time.time() < end_time:
            try:
                line = self.output_queue.get_nowait()
                output.append(line.strip())
                if self.process.poll() is not None:
                    break
            except queue.Empty:
                time.sleep(0.1)

        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()

        return "\n".join(output)


app = typer.Typer()
api_app = typer.Typer()
app.add_typer(api_app, name="api")


@api_app.command()
def ask(question: str, agent_type: str = "rag"):
    """Ask a question to the specified agent type."""
    if agent_type.lower() == "human":
        agent = HumanAgent()
    elif agent_type.lower() == "rag":
        agent = RAGAgent()
    elif agent_type.lower() == "program" and question:
        agent = ProgramWrapperAgent(question.split())
    else:
        typer.echo("Invalid agent type or missing program command.")
        return

    agent.input(question)
    response = agent.output()
    typer.echo(f"Answer: {response}")


@api_app.command()
def feed(file_path: str):
    """Feed a file to the RAG agent."""
    rag_agent = RAGAgent()
    result = rag_agent.feed_file(file_path)
    typer.echo(result)


@api_app.command()
def add_knowledge(
    knowledge: str, do_not_delete: bool = False, is_ground_truth: bool = False
):
    """Add a piece of knowledge to the RAG agent."""
    rag_agent = RAGAgent()
    rag_agent.update_knowledge_base(knowledge, do_not_delete, is_ground_truth)
    typer.echo(f"Knowledge '{knowledge}' has been added to the agent's knowledge base.")
    if do_not_delete:
        typer.echo("This knowledge is marked as 'do not delete'.")
    if is_ground_truth:
        typer.echo("This knowledge is marked as ground truth.")


@api_app.command()
def remove_knowledge(knowledge: List[str]):
    """Remove one or more pieces of knowledge from the RAG agent."""
    rag_agent = RAGAgent()
    rag_agent.remove_from_knowledge_base(knowledge)
    typer.echo(
        f"The following knowledge has been removed from the agent's knowledge base: {', '.join(knowledge)}"
    )


@api_app.command()
def remove_related(query: str, k: Optional[int] = 5):
    """Remove knowledge related to a query from the RAG agent."""
    rag_agent = RAGAgent()
    rag_agent.remove_related_memories(query, k)
    typer.echo(
        f"Knowledge related to '{query}' has been removed from the agent's knowledge base."
    )


@api_app.command()
def set_memory_protection(memory_id: int, protect: bool = True):
    """Set or unset the 'do not delete' flag for a memory."""
    rag_agent = RAGAgent()
    if rag_agent.set_memory_do_not_delete(memory_id, protect):
        action = "protected from" if protect else "allowed for"
        typer.echo(f"Memory with ID {memory_id} is now {action} deletion.")
    else:
        typer.echo(f"Memory with ID {memory_id} not found.")


@api_app.command()
def list_memories():
    """List all memories in the agent's knowledge base."""
    rag_agent = RAGAgent()
    memories = rag_agent.get_all_memories()
    for memory in memories:
        protection = "[Protected]" if memory["do_not_delete"] else ""
        typer.echo(
            f"ID: {memory['id']} {protection} - {memory['content']} ({memory['type']})"
        )


@api_app.command()
def get_relations(memory_id: int):
    """Get all relations for a specific memory."""
    rag_agent = RAGAgent()
    relations = rag_agent.get_memory_relations(memory_id)
    for relation in relations:
        typer.echo(
            f"Relation: {relation['src_id']} -> {relation['dst_id']} ({relation['relation_type']}) ({relation['type']})"
        )
        if relation["confidence"] is not None:
            typer.echo(f"Confidence: {relation['confidence']}")
        if relation["timestamp"] is not None:
            typer.echo(f"Timestamp: {relation['timestamp']}")


@api_app.command()
def identify_connections(
    k: int = 5, similarity_threshold: float = 0.5, hop_distance_threshold: int = 1
):
    """Identify connections between memories in the agent's knowledge base."""
    rag_agent = RAGAgent()
    connections = rag_agent.identify_connections(
        k, similarity_threshold, hop_distance_threshold
    )
    for connection in connections:
        typer.echo(f"Connection: {connection['connection']}")
        typer.echo(f"  Between:")
        typer.echo(f"    1. {connection['memory1']}")
        typer.echo(f"    2. {connection['memory2']}")
        typer.echo(f"  Similarity: {connection['similarity']:.2f}")
        typer.echo(f"  Hop distance: {connection['hop_distance']}")
        typer.echo(f"  Connection strength: {connection['connection_strength']:.2f}")
        typer.echo("")


@api_app.command()
def add_relation(
    src_id: int,
    dst_id: int,
    relation: str,
    confidence: Optional[float] = None,
    timestamp: Optional[str] = None,
):
    """Add a relation between two memories."""
    rag_agent = RAGAgent()
    if timestamp:
        timestamp = datetime.fromisoformat(timestamp)
    rag_agent.add_memory_relation(src_id, dst_id, relation, confidence, timestamp)
    typer.echo(f"Relation '{relation}' added between memories {src_id} and {dst_id}.")


@app.command()
def chat(agent_type: str = "rag", program: Optional[List[str]] = None):
    """Start a 1:1 chat with the specified agent type or a program wrapper."""
    if agent_type.lower() == "human":
        agent = HumanAgent()
    elif agent_type.lower() == "rag":
        agent = RAGAgent()
    elif agent_type.lower() == "program" and program:
        agent = ProgramWrapperAgent(program)
    else:
        typer.echo("Invalid agent type or missing program command.")
        return

    typer.echo(
        f"Starting a chat with {agent_type.capitalize()} Agent. Press Ctrl+C to exit."
    )

    try:
        while True:
            user_input = typer.prompt("You")
            agent.input(user_input)
            response = agent.output()
            typer.echo(f"Agent: {response}")
    except KeyboardInterrupt:
        typer.echo("\nChat ended. Goodbye!")


@app.command()
def group_chat(agent_types: List[str], programs: Optional[List[str]] = None):
    """Start a group chat with multiple agent types, including program wrappers."""
    agents = []
    for i, agent_type in enumerate(agent_types):
        if agent_type.lower() == "human":
            agents.append(HumanAgent())
        elif agent_type.lower() == "rag":
            agents.append(RAGAgent())
        elif agent_type.lower() == "program" and programs and i < len(programs):
            agents.append(ProgramWrapperAgent(programs[i].split()))
        else:
            typer.echo(
                f"Unknown agent type or missing program: {agent_type}. Skipping."
            )

    if not agents:
        typer.echo("No valid agents specified. Exiting.")
        return

    typer.echo(
        f"Starting a group chat with {len(agents)} agents. Press Ctrl+C to exit."
    )

    try:
        while True:
            user_input = typer.prompt("You")
            for i, agent in enumerate(agents):
                agent.input(user_input)
                response = agent.output()
                typer.echo(f"Agent {i+1} ({agent_types[i]}): {response}")
    except KeyboardInterrupt:
        typer.echo("\nGroup chat ended. Goodbye!")


if __name__ == "__main__":
    app()
