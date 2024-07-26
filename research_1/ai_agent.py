import os
import sys
import threading
import subprocess
import queue
import time
import uuid
from datetime import datetime
from typing import List, Optional, Any, Dict, Literal, Self, ClassVar
from textwrap import dedent

import typer
import torch
from pydantic import BaseModel, Field, UUID4
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
from anthropic import Anthropic
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# Usage
settings = Settings()

# Initialize clients
openai_client = OpenAI(api_key=settings.openai_api_key)
anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
instructor_client = instructor.from_anthropic(anthropic_client)


def embed_text(
    text: str, tokenizer: AutoTokenizer, embedding_model: AutoModel
) -> List[float]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()


def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    return 1 - cosine(embedding1, embedding2)


class BaseEntity(BaseModel):
    _memory_pool: ClassVar[dict[int, Self]] = {}
    _next_id: ClassVar[int] = 1
    id: int = Field(default_factory=lambda: BaseEntity._next_id)

    def __init_subclass__(cls, **kwargs):
        cls._memory_pool = {}
        cls._next_id = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__class__.set(self.id, self)
        self.__class__._next_id += 1

    @classmethod
    def get(cls, id: int) -> Self | None:
        return cls._memory_pool.get(id)

    @classmethod
    def create(cls, **data) -> Self:
        return cls(**data)

    @classmethod
    def update(cls, id: int, **data) -> Self | None:
        if id in cls._memory_pool:
            entity = cls._memory_pool[id]
            entity.model_copy(update=data)
            return entity
        return None

    @classmethod
    def set(cls, id: int, entity: Self) -> None:
        cls._memory_pool[id] = entity

    @classmethod
    def all(cls) -> list[Self]:
        return list(cls._memory_pool.values())

    @classmethod
    def delete(cls, id: int) -> None:
        if id in cls._memory_pool:
            del cls._memory_pool[id]


class Node(BaseEntity):
    content: str
    embedding: List[float]
    do_not_delete: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    origin: Literal["thought", "input", "output", "fact"]
    reasons: list[Node] = Field(default_factory=list)


class Relation(Node):
    src_id: int
    dst_id: int


class SyntheticRelation(Relation):
    confidence: float


class TemporalRelation(Relation):
    timestamp: datetime


class Goal(Node):
    completed: bool = False
    parent_id: Optional[int] = None
    children: List["Goal"] = []


class Agent(BaseEntity):
    type: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AIAgent(Agent):
    type: str = "ai"
    openai_model: str = "gpt-4-turbo-preview"
    anthropic_model: str = "claude-3-sonnet-20240229"
    sentence_transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_tokens: int = 300
    ultimate_goal_id: Optional[int] = None
    current_goal_id: Optional[int] = None

    _initialized: bool = False
    nodes: Dict[int, Node] = {}
    relations: Dict[int, Relation] = {}
    goals: Dict[int, Goal] = {}

    DEFAULT_INITIAL_MEMORIES = [
        "I am an experimental AI system",
        "I am continuously improving through feedback and new interactions",
    ]

    def __init__(self, **data):
        super().__init__(**data)
        self.tokenizer = AutoTokenizer.from_pretrained(self.sentence_transformer_model)
        self.embedding_model = AutoModel.from_pretrained(
            self.sentence_transformer_model
        )
        self._initialize()

    def _initialize(self):
        if self._initialized:
            return
        for knowledge in self.DEFAULT_INITIAL_MEMORIES:
            self.update_knowledge_base(knowledge)
        self._initialized = True

    def update_knowledge_base(
        self, new_info: str, do_not_delete: bool = False, is_const: bool = False
    ):
        embedding = self.embed_text(new_info)
        node = Node.create(
            content=new_info,
            embedding=embedding,
            do_not_delete=do_not_delete,
            agent_id=self.id,
        )
        self.nodes[node.id] = node

    def remove_from_knowledge_base(self, info_to_remove: List[str]):
        for node_id, node in list(self.nodes.items()):
            if node.content in info_to_remove and not node.do_not_delete:
                del self.nodes[node_id]

    def remove_related_nodes(self, query: str, k: int = 5):
        query_embedding = self.embed_text(query)
        related_nodes = sorted(
            self.nodes.values(),
            key=lambda n: self.calculate_similarity(n.embedding, query_embedding),
            reverse=True,
        )[:k]
        for node in related_nodes:
            if not node.do_not_delete:
                del self.nodes[node.id]

    def set_node_do_not_delete(self, node_id: int, do_not_delete: bool = True):
        if node_id in self.nodes:
            self.nodes[node_id].do_not_delete = do_not_delete
            return True
        return False

    def get_all_nodes(self) -> List[dict]:
        return [
            {
                "id": n.id,
                "content": n.content,
                "do_not_delete": n.do_not_delete,
                "type": type(n).__name__,
                "agent_id": n.agent_id,
            }
            for n in self.nodes.values()
        ]

    def get_node_relations(self, node_id: int) -> List[dict]:
        return [
            {
                "src_id": r.src_id,
                "dst_id": r.dst_id,
                "relation_type": r.content,
                "type": type(r).__name__,
                "confidence": getattr(r, "confidence", None),
                "timestamp": getattr(r, "timestamp", None),
            }
            for r in self.relations.values()
            if r.src_id == node_id or r.dst_id == node_id
        ]

    def identify_connections(
        self,
        k: int = 5,
        similarity_threshold: float = 0.5,
        hop_distance_threshold: int = 3,
    ):
        connections = []
        for node1 in self.nodes.values():
            for node2 in self.nodes.values():
                if node1.id != node2.id:
                    similarity = self.calculate_similarity(
                        node1.embedding, node2.embedding
                    )
                    hop_distance = self.calculate_hop_distance(node1.id, node2.id)
                    if (
                        similarity > similarity_threshold
                        and hop_distance <= hop_distance_threshold
                    ):
                        connection = self.describe_connection(
                            node1.content, node2.content
                        )
                        connections.append(
                            {
                                "connection": connection,
                                "node1": node1.content,
                                "node2": node2.content,
                                "similarity": similarity,
                                "hop_distance": hop_distance,
                                "connection_strength": similarity
                                * (1.0 / max(1, hop_distance)),
                            }
                        )
        return sorted(
            connections, key=lambda x: x["connection_strength"], reverse=True
        )[:k]

    def describe_connection(self, content1: str, content2: str) -> str:
        prompt = f"Given these two pieces of information:\n1. {content1}\n2. {content2}\n\nDescribe the connection or relationship between these two pieces of information."

        response = openai_client.chat.completions.create(
            model=self.openai_model,
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

    def calculate_hop_distance(self, start_id: int, end_id: int) -> int:
        visited = set()
        queue = [(start_id, 0)]

        while queue:
            current_id, distance = queue.pop(0)

            if current_id == end_id:
                return distance

            if current_id not in visited:
                visited.add(current_id)
                for relation in self.relations.values():
                    if relation.src_id == current_id and relation.dst_id not in visited:
                        queue.append((relation.dst_id, distance + 1))
                    elif (
                        relation.dst_id == current_id and relation.src_id not in visited
                    ):
                        queue.append((relation.src_id, distance + 1))

        return -1  # No path found

    def get_relevant_nodes(
        self,
        node_id: int,
        k: int = 5,
        similarity_weight: float = 0.5,
        recency_weight: float = 0.2,
        hop_distance_weight: float = 0.3,
    ) -> List[tuple[Node, float]]:
        target_node = self.nodes[node_id]
        relevant_nodes = []

        for other_node in self.nodes.values():
            if other_node.id != node_id:
                similarity = self.calculate_similarity(
                    target_node.embedding, other_node.embedding
                )
                recency = 1.0 / (
                    1
                    + (datetime.utcnow() - other_node.created_at).total_seconds() / 3600
                )
                hop_distance = self.calculate_hop_distance(node_id, other_node.id)

                relevance = (
                    similarity * similarity_weight
                    + recency * recency_weight
                    + (1.0 / max(1, hop_distance)) * hop_distance_weight
                )

                relevant_nodes.append((other_node, relevance))

        return sorted(relevant_nodes, key=lambda x: x[1], reverse=True)[:k]

    def input(self, data: str, **kwargs):
        if not isinstance(data, str):
            raise ValueError("Input data must be a string")

        input_node = Node.create(
            content=data,
            embedding=self.embed_text(data),
            agent_id=self.id,
        )
        self.nodes[input_node.id] = input_node

        relevant_nodes = [
            node.content
            for node, _ in self.get_relevant_nodes(
                node_id=input_node.id,
                k=5,
                similarity_weight=0.5,
                recency_weight=0.2,
                hop_distance_weight=0.3,
            )
        ]

        extracted_nodes_response = instructor_client.messages.create(
            model=self.anthropic_model,
            max_tokens=4096,
            messages=[
                {
                    "role": "system",
                    "content": "Extract a list of relevant nodes from the given context and user input.",
                },
                {
                    "role": "user",
                    "content": dedent(
                        f"""
                        Input:
                        <input>
                        {data}
                        </input>

                        Related:
                        <related>
                        {relevant_nodes}
                        </related>
                        
                        Extract a list of additional relevant nodes:
                        <additional_relevant>
                        """
                    ).strip(),
                },
            ],
            response_model=List[str],
            stop=["</additional_relevant>"],
        )
        extracted_node_contents = extracted_nodes_response.choices[
            0
        ].message.content.split("\n")
        extracted_nodes = [
            Node.create(
                content=node_content,
                embedding=self.embed_text(node_content),
                agent_id=self.id,
                origin="thought",
                reasons=[input_node],
            )
            for node_content in extracted_node_contents
        ]
        connect(self.latent_nodes, extracted_nodes)

        # Generate response using Claude 3.5 Sonnet
        response = anthropic_client.messages.create(
            model=self.anthropic_model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {
                    "role": "user",
                    "content": f"Relevant nodes: {' | '.join(relevant_context)}\n\nMost recent input: {recent_input.content}",
                },
            ],
        )
        return response.content[0].text


class HumanAgent(Agent):
    type: str = "human"
    last_query: Optional[str] = None

    def input(self, data: Any, **kwargs):
        self.last_query = data if isinstance(data, str) else str(data)
        print(f"\nQuestion: {self.last_query}")

    def output(self, **kwargs) -> Any:
        if not self.last_query:
            raise ValueError("Input must be called before output")
        print("Please provide your response:")
        return sys.stdin.readline().strip()


class LineReaderBot(Agent):
    type: str = "line_reader"
    file_path: str
    current_line: int = 0
    total_lines: int
    file: Any  # We can't type hint file objects directly

    def __init__(self, **data):
        super().__init__(**data)
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
    type: str = "program"
    command: List[str]
    timeout: float = 5.0
    process: Any  # subprocess.Popen object
    output_queue: queue.Queue
    last_input: Optional[str] = None

    def input(self, data: Any, **kwargs):
        self.last_input = data

    def output(self, **kwargs) -> Any:
        if not self.last_input:
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

        self.output_queue = queue.Queue()

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


def get_agent(agent_id: str):
    if agent_id.lower() == "human":
        return HumanAgent(id=agent_id)
    elif agent_id.lower() == "program":
        program = typer.prompt("Enter the program invocation command")
        program = program.split()
        return ProgramWrapperAgent(id=agent_id, command=program)
    else:
        return AIAgent(id=agent_id)


app = typer.Typer()
chat_app = typer.Typer()
api_app = typer.Typer()

app.add_typer(chat_app, name="chat")
app.add_typer(api_app, name="api")


def start_chat(agents: List[Any], chat_type: str):
    """Start a chat with one or more agents."""
    if not agents:
        typer.echo("No agents specified. Exiting.")
        return

    agent_count = len(agents)
    typer.echo(
        f"Starting a {chat_type} chat with {agent_count} agent{'s' if agent_count > 1 else ''}. Press Ctrl+C to exit."
    )

    try:
        last_statement: str = input("You: ")
        while True:
            for i, agent in enumerate(agents):
                agent.input(last_statement)
                response = agent.output()
                if agent_count == 1:
                    typer.echo(f"Agent: {response}")
                else:
                    typer.echo(f"Agent {i+1} ({agent.__class__.__name__}): {response}")
                last_statement = response
    except KeyboardInterrupt:
        typer.echo(f"\n{chat_type.capitalize()} chat ended. Goodbye!")


@chat_app.command("one-on-one")
def chat_one_on_one(agent_id: str):
    """Start a 1:1 chat with the specified agent."""
    agent = get_agent(agent_id)
    human_agent = HumanAgent(id="human")
    start_chat([human_agent, agent], "one-on-one")


@chat_app.command("group")
def chat_group(*agent_ids: str):
    """Start a group chat with multiple agent types."""
    agents = [get_agent(agent_id) for agent_id in agent_ids]
    start_chat(agents, "group")


@api_app.command()
def ask(agent_id: str, question: str):
    """Ask a question to the specified agent."""
    agent = get_agent(agent_id)
    agent.input(question)
    response = agent.output()
    typer.echo(f"Answer: {response}")


@api_app.command()
def add_knowledge(
    agent_id: str,
    knowledge: str,
    do_not_delete: bool = False,
    is_const: bool = False,
):
    """Add a piece of knowledge to the specified agent."""
    agent = get_agent(agent_id)
    if isinstance(agent, AIAgent):
        agent.update_knowledge_base(knowledge, do_not_delete, is_const)
        typer.echo(
            f"Knowledge '{knowledge}' has been added to the agent's knowledge base."
        )
        if do_not_delete:
            typer.echo("This knowledge is marked as 'do not delete'.")
        if is_const:
            typer.echo("This knowledge is marked as constant.")
    else:
        typer.echo(f"Agent '{agent_id}' does not support adding knowledge.")


@api_app.command()
def remove_knowledge(agent_id: str, knowledge: List[str]):
    """Remove one or more pieces of knowledge from the specified agent."""
    agent = get_agent(agent_id)
    if isinstance(agent, AIAgent):
        agent.remove_from_knowledge_base(knowledge)
        typer.echo(
            f"The following knowledge has been removed from the agent's knowledge base: {', '.join(knowledge)}"
        )
    else:
        typer.echo(f"Agent '{agent_id}' does not support removing knowledge.")


@api_app.command()
def remove_related(agent_id: str, query: str, k: Optional[int] = 5):
    """Remove knowledge related to a query from the specified agent."""
    agent = get_agent(agent_id)
    if isinstance(agent, AIAgent):
        agent.remove_related_nodes(query, k)
        typer.echo(
            f"Knowledge related to '{query}' has been removed from the agent's knowledge base."
        )
    else:
        typer.echo(f"Agent '{agent_id}' does not support removing related knowledge.")


@api_app.command()
def set_node_protection(agent_id: str, node_id: int, protect: bool = True):
    """Set or unset the 'do not delete' flag for a node in the specified agent."""
    agent = get_agent(agent_id)
    if isinstance(agent, AIAgent):
        if agent.set_node_do_not_delete(node_id, protect):
            action = "protected from" if protect else "allowed for"
            typer.echo(f"Node with ID {node_id} is now {action} deletion.")
        else:
            typer.echo(f"Node with ID {node_id} not found.")
    else:
        typer.echo(f"Agent '{agent_id}' does not support node protection.")


@api_app.command()
def list_nodes(agent_id: str):
    """List all nodes in the specified agent's knowledge base."""
    agent = get_agent(agent_id)
    if isinstance(agent, AIAgent):
        nodes = agent.get_all_nodes()
        for node in nodes:
            protection = "[Protected]" if node["do_not_delete"] else ""
            typer.echo(
                f"ID: {node['id']} {protection} - {node['content']} ({node['type']})"
            )
    else:
        typer.echo(f"Agent '{agent_id}' does not support listing nodes.")


@api_app.command()
def get_relations(agent_id: str, node_id: int):
    """Get all relations for a specific node in the specified agent."""
    agent = get_agent(agent_id)
    if isinstance(agent, AIAgent):
        relations = agent.get_node_relations(node_id)
        for relation in relations:
            typer.echo(
                f"Relation: {relation['src_id']} -> {relation['dst_id']} ({relation['relation_type']}) ({relation['type']})"
            )
            if relation["confidence"] is not None:
                typer.echo(f"Confidence: {relation['confidence']}")
            if relation["timestamp"] is not None:
                typer.echo(f"Timestamp: {relation['timestamp']}")
    else:
        typer.echo(f"Agent '{agent_id}' does not support getting relations.")


@api_app.command()
def identify_connections(
    agent_id: str,
    k: int = 5,
    similarity_threshold: float = 0.5,
    hop_distance_threshold: int = 1,
):
    """Identify connections between nodes in the specified agent's knowledge base."""
    agent = get_agent(agent_id)
    if isinstance(agent, AIAgent):
        connections = agent.identify_connections(
            k, similarity_threshold, hop_distance_threshold
        )
        for connection in connections:
            typer.echo(f"Connection: {connection['connection']}")
            typer.echo(f"  Between:")
            typer.echo(f"    1. {connection['node1']}")
            typer.echo(f"    2. {connection['node2']}")
            typer.echo(f"  Similarity: {connection['similarity']:.2f}")
            typer.echo(f"  Hop distance: {connection['hop_distance']}")
            typer.echo(
                f"  Connection strength: {connection['connection_strength']:.2f}"
            )
            typer.echo("")
    else:
        typer.echo(f"Agent '{agent_id}' does not support identifying connections.")


if __name__ == "__main__":
    app()
