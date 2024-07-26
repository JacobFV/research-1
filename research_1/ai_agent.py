# Standard library imports
import os
import sys
import threading
import subprocess
import queue
import time
from datetime import datetime
from abc import ABC, abstractmethod
from typing import List, Optional, Any, Tuple
from textwrap import dedent

# Third-party imports
import typer
import torch
from pydantic import BaseModel
from scipy.spatial.distance import cosine
from sqlalchemy import text
from sqlmodel import Field, Session, SQLModel, create_engine, select, Relationship
from pgvector.sqlalchemy import Vector
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
from anthropic import Anthropic
from pydantic_settings import BaseSettings, SettingsConfigDict

# Local imports (if any)
# import custom_module


class DatabaseSettings(BaseSettings):
    host: str = "localhost"
    port: int = 5432
    name: str = "ragagent"
    user: str = "user"
    password: str = "password"

    model_config = SettingsConfigDict(env_prefix="DB_")

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class Settings(BaseSettings):
    database: DatabaseSettings = DatabaseSettings()
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


class BaseSQLModel(SQLModel, table=False):
    @classmethod
    def _cls_initialize(cls):
        cls._create_stored_procedures()

    @classmethod
    def _create_stored_procedures(cls):
        pass


# Usage
settings = Settings()

# Initialize clients
openai_client = OpenAI(api_key=settings.openai_api_key)
anthropic_client = Anthropic(api_key=settings.anthropic_api_key)

# Global engine
engine = create_engine(settings.database.url)


def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """Split the input text into chunks of approximately equal size."""
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def embed_text(text: str, tokenizer, embedding_model) -> List[float]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()


def parse_model(model: type[BaseModel], prompt: str, **kwargs) -> BaseModel:
    """Parse the response using the Instructor library."""
    return openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=model,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )


class Node(BaseSQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    content: str
    embedding: List[float] = Field(sa_column=Vector(384))
    do_not_delete: bool = Field(default=False)
    agent_id: str = Field(default="default")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    __table_args__ = {"extend_existing": True}

    # Add back-references
    outgoing_relations: List["Relation"] = Relationship(back_populates="src")
    incoming_relations: List["Relation"] = Relationship(back_populates="dst")


class ConstNode(Node, table=True):
    __tablename__ = "const_node"


class InputNode(Node, table=True):
    __tablename__ = "input_node"


class ThoughtNode(Node, table=True):
    __tablename__ = "thought_node"


class LogEntryNode(Node, table=True):
    __tablename__ = "log_entry_node"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Relation(Node, table=True):
    __tablename__ = "relation"
    src_id: int = Field(foreign_key="abstract_node.id")
    dst_id: int = Field(foreign_key="abstract_node.id")

    # Add relations
    src: "Node" = Relationship(
        back_populates="outgoing_relations",
        sa_relationship_kwargs={"foreign_keys": [src_id]},
    )
    dst: "Node" = Relationship(
        back_populates="incoming_relations",
        sa_relationship_kwargs={"foreign_keys": [dst_id]},
    )


class SyntheticRelation(Relation, table=True):
    __tablename__ = "synthetic_relation"
    confidence: float


class TemporalRelation(Relation, table=True):
    __tablename__ = "temporal_relation"
    timestamp: datetime


class Goal(Node, table=True):
    __tablename__ = "goal"

    # Inherited fields from Node:
    # id: Optional[int] = Field(default=None, primary_key=True)
    # content: str  # This will store the goal description
    # embedding: List[float] = Field(sa_column=Vector(384))
    # do_not_delete: bool = Field(default=False)
    # agent_id: str = Field(default="default")
    # created_at: datetime = Field(default_factory=datetime.utcnow)

    # Goal-specific fields:
    completed: bool = Field(default=False)
    # TODO: somehow i need to make these parent and children fields work with the incoming/outgoing relations field. They need to be getters/setters and init args
    parent_id: Optional[int] = Field(default=None, foreign_key="goal.id")

    # Relationships
    children: List["Goal"] = Relationship(back_populates="parent")
    parent: Optional["Goal"] = Relationship(
        back_populates="children", link_column="parent_id"
    )
    agent: "AIAgent" = Relationship(back_populates="goals")


class Agent(BaseSQLModel, table=True):
    id: str = Field(primary_key=True)
    type: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AIAgent(Agent, table=True):
    __tablename__ = "ai_agent"

    # Existing fields from Agent class
    id: str = Field(primary_key=True)
    type: str = Field(default="ai")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # New fields specific to AI
    openai_model: str = Field(default="gpt-4-turbo-preview")
    anthropic_model: str = Field(default="claude-3-sonnet-20240229")
    sentence_transformer_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2"
    )
    max_tokens: int = Field(default=300)

    # New goal-related fields
    ultimate_goal_id: Optional[int] = Field(default=None, foreign_key="goal.id")
    current_goal_id: Optional[int] = Field(default=None, foreign_key="goal.id")

    # Relationships
    goals: List[Goal] = Relationship(back_populates="agent")
    ultimate_goal: Optional[Goal] = Relationship(link_column="ultimate_goal_id")
    current_goal: Optional[Goal] = Relationship(link_column="current_goal_id")

    # New class variable for default initial memories
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

    @property
    def is_initialized(self):
        with Session(engine) as session:
            return session.exec(select(Node)).first() is not None

    def _initialize(self):
        if self.is_initialized:
            return
        for knowledge in self.DEFAULT_INITIAL_MEMORIES:
            self.update_knowledge_base(knowledge)

    @classmethod
    def _cls_initialize(cls):
        cls._create_stored_procedures()

    @classmethod
    def _create_stored_procedures(cls):
        with engine.connect() as connection:
            connection.execute(
                text(
                    dedent(
                        """\
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
                                    -- Check outgoing relations
                                    FOR relation IN SELECT dst_id FROM relation WHERE src_id = current_id LOOP
                                        IF NOT (relation.dst_id = ANY(visited)) THEN
                                            queue := array_cat(queue, ARRAY[[relation.dst_id, distance + 1]]::INTEGER[][]);
                                        END IF;
                                    END LOOP;
                                    -- Check incoming relations
                                    FOR relation IN SELECT src_id FROM relation WHERE dst_id = current_id LOOP
                                        IF NOT (relation.src_id = ANY(visited)) THEN
                                            queue := array_cat(queue, ARRAY[[relation.src_id, distance + 1]]::INTEGER[][]);
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

                        -- Function for computing relevant nodes
                        CREATE OR REPLACE FUNCTION compute_relevant_nodes(
                            p_node_id INTEGER,
                            p_k INTEGER,
                            p_agent_id TEXT,
                            p_similarity_weight FLOAT DEFAULT 0.5,
                            p_recency_weight FLOAT DEFAULT 0.2,
                            p_hop_distance_weight FLOAT DEFAULT 0.3
                        )
                        RETURNS TABLE (
                            rank INTEGER,
                            relevant_node_id INTEGER,
                            relevance FLOAT
                        ) AS $$
                        DECLARE
                            context_embedding FLOAT[];
                            total_weight FLOAT;
                        BEGIN
                            -- Validate weights
                            total_weight := p_similarity_weight + p_recency_weight + p_hop_distance_weight;
                            IF total_weight != 1.0 THEN
                                RAISE EXCEPTION 'Weights must sum to 1.0. Current sum: %', total_weight;
                            END IF;

                            -- Get the embedding for the given node_id
                            SELECT embedding INTO context_embedding
                            FROM abstract_node
                            WHERE id = p_node_id AND agent_id = p_agent_id;

                            -- If no embedding found, return empty result
                            IF context_embedding IS NULL THEN
                                RETURN;
                            END IF;

                            RETURN QUERY
                            WITH relevance_calc AS (
                                SELECT 
                                    n.id AS relevant_node_id,
                                    (1 - cosine_distance(n.embedding, context_embedding)) * p_similarity_weight + 
                                    (1.0 / (1 + EXTRACT(EPOCH FROM (NOW() - n.created_at)) / 3600)) * p_recency_weight +
                                    (
                                        CASE 
                                            WHEN EXISTS (
                                                SELECT 1 
                                                FROM synthetic_relation srt 
                                                WHERE (srt.src_id = p_node_id AND srt.dst_id = n.id) 
                                                   OR (srt.dst_id = p_node_id AND srt.src_id = n.id)
                                            ) THEN
                                                (1.0 / (1 + COALESCE(calculate_hop_distance(p_node_id, n.id), 100))) * 
                                                (
                                                    SELECT COALESCE(AVG(srt.confidence), 1.0)
                                                    FROM synthetic_relation srt 
                                                    WHERE (srt.src_id = p_node_id AND srt.dst_id = n.id) 
                                                       OR (srt.dst_id = p_node_id AND srt.src_id = n.id)
                                                )
                                            ELSE
                                                1.0 / (1 + COALESCE(calculate_hop_distance(p_node_id, n.id), 100))
                                        END
                                    ) * p_hop_distance_weight AS relevance
                                FROM abstract_node n
                                WHERE n.agent_id = p_agent_id AND n.id != p_node_id
                            )
                            SELECT 
                                row_number() OVER (ORDER BY relevance DESC) AS rank,
                                relevant_node_id,
                                relevance
                            FROM relevance_calc
                            ORDER BY relevance DESC
                            LIMIT p_k;
                        END;
                        $$ LANGUAGE plpgsql;
                        """
                    )
                )
            )
            connection.commit()

    def update_knowledge_base(
        self, new_info: str, do_not_delete: bool = False, is_const: bool = False
    ):
        embedding = embed_text(new_info, self.tokenizer, self.embedding_model)
        if is_const:
            node = ConstNode(
                content=new_info,
                embedding=embedding,
                do_not_delete=do_not_delete,
                agent_id=self.id,
            )
        else:
            node = Node(
                content=new_info,
                embedding=embedding,
                do_not_delete=do_not_delete,
                agent_id=self.id,
            )
        with Session(engine) as session:
            session.add(node)
            session.commit()

    def add_node_relation(
        self,
        src_id: int,
        dst_id: int,
        relation: str,
        confidence: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ):
        with Session(engine) as session:
            if confidence is not None:
                node_relation = SyntheticRelation(
                    content=relation,
                    src_id=src_id,
                    dst_id=dst_id,
                    confidence=confidence,
                )
            elif timestamp is not None:
                node_relation = TemporalRelation(
                    content=relation,
                    src_id=src_id,
                    dst_id=dst_id,
                    timestamp=timestamp,
                )
            else:
                node_relation = Relation(content=relation, src_id=src_id, dst_id=dst_id)
            session.add(node_relation)
            session.commit()

    def calculate_hop_distance(self, start_id: int, end_id: int) -> int:
        with engine.connect() as connection:
            result = connection.execute(
                text(
                    """
                    SELECT calculate_hop_distance(:start_id, :end_id)
                    WHERE EXISTS (
                        SELECT 1 FROM abstract_node
                        WHERE id IN (:start_id, :end_id) AND agent_id = :agent_id
                    )
                """
                ),
                {"start_id": start_id, "end_id": end_id, "agent_id": self.id},
            )
            return result.scalar_one()

    def identify_connections(
        self,
        k: int = 5,
        similarity_threshold: float = 0.5,
        hop_distance_threshold: int = 3,
    ):
        with engine.connect() as connection:
            result = connection.execute(
                text(
                    """
                    WITH node_pairs AS (
                        SELECT 
                            m1.id AS node1_id,
                            m1.content AS node1_content,
                            m2.id AS node2_id,
                            m2.content AS node2_content,
                            1 - cosine_distance(m1.embedding, m2.embedding) AS similarity,
                            calculate_hop_distance(m1.id, m2.id) AS hop_distance
                        FROM abstract_node m1
                        CROSS JOIN abstract_node m2
                        WHERE m1.id != m2.id
                        AND m1.agent_id = :agent_id
                        AND m2.agent_id = :agent_id
                    )
                    SELECT 
                        node1_content,
                        node2_content,
                        similarity,
                        hop_distance,
                        similarity * (1.0 / NULLIF(hop_distance, 0)) AS connection_strength
                    FROM node_pairs
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
                    "agent_id": self.id,
                },
            )

            connections = []
            for row in result:
                connection = self.describe_connection(
                    row.node1_content, row.node2_content
                )
                connections.append(
                    {
                        "connection": connection,
                        "node1": row.node1_content,
                        "node2": row.node2_content,
                        "similarity": row.similarity,
                        "hop_distance": row.hop_distance,
                        "connection_strength": row.connection_strength,
                    }
                )
            return connections

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

    def calculate_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        return 1 - cosine(embedding1, embedding2)

    def remove_from_knowledge_base(
        self, info_to_remove: List[str], agent_id: Optional[str] = None
    ):
        with Session(engine) as session:
            for info in info_to_remove:
                query = select(Node).where(
                    Node.content == info,
                    Node.do_not_delete == False,
                )
                if agent_id:
                    query = query.where(Node.agent_id == agent_id)
                node = session.exec(query).first()
                if node:
                    session.delete(node)
            session.commit()

    def remove_related_nodes(
        self, query: str, k: int = 5, agent_id: Optional[str] = None
    ):
        embedding = embed_text(query, self.tokenizer, self.embedding_model)
        with Session(engine) as session:
            query = select(Node).where(Node.do_not_delete == False)
            if agent_id:
                query = query.where(Node.agent_id == agent_id)
            related_nodes = session.exec(
                query.order_by(Node.embedding.cosine_distance(embedding)).limit(k)
            ).all()
            for node in related_nodes:
                session.delete(node)
            session.commit()

    def set_node_do_not_delete(self, node_id: int, do_not_delete: bool = True):
        with Session(engine) as session:
            node = session.get(Node, node_id)
            if node:
                node.do_not_delete = do_not_delete
                session.add(node)
                session.commit()
                return True
            return False

    def get_all_nodes(self, agent_id: Optional[str] = None) -> List[dict]:
        with Session(engine) as session:
            query = select(Node)
            if agent_id:
                query = query.where(Node.agent_id == agent_id)
            nodes = session.exec(query).all()
            return [
                {
                    "id": n.id,
                    "content": n.content,
                    "do_not_delete": n.do_not_delete,
                    "type": type(n).__name__,
                    "agent_id": n.agent_id,
                }
                for n in nodes
            ]

    def get_node_relations(self, node_id: int) -> List[dict]:
        with Session(engine) as session:
            relations = session.exec(
                select(Relation).where(
                    (Relation.src_id == node_id) | (Relation.dst_id == node_id)
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

    def get_relevant_nodes(
        self,
        node_id: int,
        k: int = 5,
        similarity_weight: float = 0.5,
        recency_weight: float = 0.2,
        hop_distance_weight: float = 0.3,
    ) -> List[Tuple[Node, float]]:

        with Session(engine) as session:
            result = session.exec(
                select(Node, text("relevance"))
                .from_statement(
                    text(
                        dedent(
                            """\
                            SELECT n.*, r.relevance 
                            FROM compute_relevant_nodes(
                                :node_id, 
                                :k, 
                                :agent_id, 
                                :similarity_weight, 
                                :recency_weight, 
                                :hop_distance_weight
                            ) r 
                            JOIN abstract_node n ON r.relevant_node_id = n.id
                        """
                        )
                    )
                )
                .params(
                    node_id=node_id,
                    k=k,
                    agent_id=self.id,
                    similarity_weight=similarity_weight,
                    recency_weight=recency_weight,
                    hop_distance_weight=hop_distance_weight,
                )
            )

            relevant_nodes = []
            for node, relevance in result:
                relevant_nodes.append((node, relevance))

            return relevant_nodes

    def input(self, data: str, **kwargs):
        if not isinstance(data, str):
            raise ValueError("Input data must be a string")

        with Session(engine) as session:
            # Store InputNode
            input_node = InputNode(
                content=data,
                embedding=embed_text(data, self.tokenizer, self.embedding_model),
                agent_id=self.id,
            )
            session.add(input_node)
            session.commit()

        # Get relevant nodes
        relevant_nodes = [
            node.content
            for node, _ in self.get_relevant_nodes(
                node_id=input_node.id,
                k=5,  # You can adjust this number as needed
                similarity_weight=0.5,
                recency_weight=0.2,
                hop_distance_weight=0.3,
            )
        ]

        # Extract relevant nodes
        extracted_nodes_response = openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": "Extract a list of relevant nodes from the given context and user input.",
                },
                {
                    "role": "user",
                    "content": dedent(
                        f"""
                        Current context:
                        <context>
                        {data}
                        </context>

                        Relevant nodes:
                        <relevant_nodes>
                        {relevant_nodes}
                        </relevant_nodes>

                        Last input:
                        <last_input>
                        {data}
                        </last_input>

                        Extract a list of relevant nodes:
                        """
                    ).strip(),
                },
            ],
            max_tokens=150,
        )
        extracted_nodes = extracted_nodes_response.choices[0].message.content.split(
            "\n"
        )

        # Add extracted nodes to the database
        with Session(engine) as session:
            for node_content in extracted_nodes:
                # Create embedding for the node
                node_embedding = embed_text(
                    node_content, self.tokenizer, self.embedding_model
                )

                # Create new ThoughtNode object
                new_node = ThoughtNode(
                    content=node_content,
                    embedding=node_embedding,
                    agent_id=self.id,
                )

                # Add the new node to the session
                session.add(new_node)

                # Create a relation between the new synthetic node and the input node
                relation = SyntheticRelation(
                    content="Derived from",
                    src_id=new_node.id,
                    dst_id=input_node.id,
                    agent_id=self.id,
                    confidence=0.8,  # You can adjust this confidence value as needed
                )
                session.add(relation)

            # Commit the changes to the database
            session.commit()

    def output(self, **kwargs) -> Any:
        # Get relevant nodes
        relevant_nodes = self.get_relevant_nodes(
            node_id=input_node.id,
            k=5,  # You can adjust this number as needed
            similarity_weight=0.5,
            recency_weight=0.2,
            hop_distance_weight=0.3,
        )
        relevant_context = [
            f"{node.content} (relevance: {relevance:.2f})"
            for node, relevance in relevant_nodes
        ]

        # Generate response using Claude 3.5 Sonnet
        response = anthropic_client.messages.create(
            model=self.anthropic_model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {
                    "role": "user",
                    "content": f"Relevant nodes: {' | '.join(relevant_context)}\n\nExtracted nodes: {' | '.join(self.extracted_nodes)}\n\nMost recent input: {recent_input.content if recent_input else 'No recent input'}",
                },
            ],
        )
        return response.content[0].text

    def set_ultimate_goal(self, goal_content: str):
        with Session(engine) as session:
            ultimate_goal = Goal(content=goal_content, agent_id=self.id)
            session.add(ultimate_goal)
            session.commit()
            self.ultimate_goal_id = ultimate_goal.id
            session.add(self)
            session.commit()

    def set_current_goal(self, goal_id: int):
        with Session(engine) as session:
            goal = session.get(Goal, goal_id)
            if goal and goal.agent_id == self.id:
                self.current_goal_id = goal_id
                session.add(self)
                session.commit()
            else:
                raise ValueError(
                    "Invalid goal_id or goal does not belong to this agent"
                )

    def add_subgoal(self, parent_goal_id: int, subgoal_content: str):
        with Session(engine) as session:
            parent_goal = session.get(Goal, parent_goal_id)
            if parent_goal and parent_goal.agent_id == self.id:
                subgoal = Goal(
                    content=subgoal_content,
                    parent_id=parent_goal_id,
                    agent_id=self.id,
                )
                session.add(subgoal)
                session.commit()
                return subgoal.id
            else:
                raise ValueError(
                    "Invalid parent_goal_id or goal does not belong to this agent"
                )

    def complete_goal(self, goal_id: int):
        with Session(engine) as session:
            goal = session.get(Goal, goal_id)
            if goal and goal.agent_id == self.id:
                goal.completed = True
                session.add(goal)
                session.commit()
            else:
                raise ValueError(
                    "Invalid goal_id or goal does not belong to this agent"
                )

    def get_goal_hierarchy(self) -> dict:
        with Session(engine) as session:
            ultimate_goal = session.get(Goal, self.ultimate_goal_id)
            if not ultimate_goal:
                return {}

            def build_goal_tree(goal):
                return {
                    "id": goal.id,
                    "content": goal.content,
                    "completed": goal.completed,
                    "children": [build_goal_tree(child) for child in goal.children],
                }

            return build_goal_tree(ultimate_goal)


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


def get_agent(agent_id: str):
    if agent_id.lower() == "human":
        return HumanAgent()
    elif agent_id.lower() == "program":
        program = typer.prompt("Enter the program invocation command")
        program = program.split()
        return ProgramWrapperAgent(program)
    else:
        # RAG agent creation logic (previously in get_or_create)
        with Session(engine) as session:
            agent = session.get(Agent, agent_id)
            if not agent:
                agent = AIAgent(id=agent_id)
                session.add(agent)
                session.commit()
        return agent


app = typer.Typer()
chat_app = typer.Typer()
api_app = typer.Typer()
config_app = typer.Typer()

app.add_typer(chat_app, name="chat")
app.add_typer(api_app, name="api")
app.add_typer(config_app, name="config")


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
    if agent_id.lower() not in ["human", "program"]:
        confirm = typer.confirm(
            f"Agent '{agent_id}' not found. Create a new RAG agent with this ID?"
        )
        if not confirm:
            typer.echo("Chat cancelled.")
            return

    agent = get_agent(agent_id)
    human_agent = HumanAgent()
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
def feed(agent_id: str, file_path: str):
    """Feed a file to the specified agent."""
    agent = get_agent(agent_id)
    if isinstance(agent, AIAgent):
        result = agent.feed_file(file_path)
        typer.echo(result)
    else:
        typer.echo(f"Agent '{agent_id}' does not support file feeding.")


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


@api_app.command()
def add_relation(
    agent_id: str,
    src_id: int,
    dst_id: int,
    relation: str,
    confidence: Optional[float] = None,
    timestamp: Optional[str] = None,
):
    """Add a relation between two nodes in the specified agent."""
    agent = get_agent(agent_id)
    if isinstance(agent, AIAgent):
        if timestamp:
            timestamp = datetime.fromisoformat(timestamp)
        agent.add_node_relation(src_id, dst_id, relation, confidence, timestamp)
        typer.echo(f"Relation '{relation}' added between nodes {src_id} and {dst_id}.")
    else:
        typer.echo(f"Agent '{agent_id}' does not support adding relations.")


def is_initialized(app: typer.Typer) -> bool:
    """Check if the application has been initialized."""
    init_flag_file = os.path.join(app.info.app_dir, ".initialized")
    return os.path.exists(init_flag_file)


def mark_as_initialized(app: typer.Typer):
    """Mark the application as initialized."""
    init_flag_file = os.path.join(app.info.app_dir, ".initialized")
    os.makedirs(app.info.app_dir, exist_ok=True)
    with open(init_flag_file, "w") as f:
        f.write("initialized")


@config_app.command("init")
def init_db():
    """Initialize the database schema and stored procedures."""
    if is_initialized(app):
        typer.echo("Application is already initialized.")
        reinit = typer.confirm("Do you want to reinitialize?")
        if not reinit:
            typer.echo("Initialization cancelled.")
            return

    typer.echo("Initializing database schema and stored procedures...")
    first_init()
    mark_as_initialized(app)
    typer.echo("Database schema and stored procedures initialized successfully.")


@config_app.command("clean")
def clean_db():
    """Clean all data from the database."""
    confirm = typer.confirm("This will delete all data in the database. Are you sure?")
    if not confirm:
        typer.echo("Operation cancelled.")
        return

    typer.echo("Cleaning database...")
    with Session(engine) as session:
        # Delete all data from all tables
        for table in reversed(SQLModel.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()
    typer.echo("Database cleaned successfully.")


def first_init():
    """Initialize the database and all BaseSQLModel subclasses."""
    SQLModel.metadata.create_all(engine)
    for cls in BaseSQLModel.__subclasses__():
        cls._cls_initialize()


if not is_initialized(app):
    typer.echo("First-time initialization...")
    first_init()
    mark_as_initialized(app)
    typer.echo("Initialization complete.")
else:
    typer.echo("Application already initialized. Skipping first-time setup.")

if __name__ == "__main__":
    app()
