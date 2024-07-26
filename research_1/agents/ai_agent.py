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

# Third-party imports
import typer
import torch
from pydantic import BaseModel
from scipy.spatial.distance import cosine
from sqlalchemy import text
from sqlmodel import Field, Session, SQLModel, create_engine, select
from pgvector.sqlalchemy import Vector
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
from anthropic import Anthropic

# Local imports (if any)
# import custom_module

# Initialize clients
openai_client = OpenAI()
anthropic_client = Anthropic()
