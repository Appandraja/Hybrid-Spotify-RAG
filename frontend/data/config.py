import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
GRAPH_STORE_PATH = os.getenv("GRAPH_STORE_PATH", "./data/knowledge_graph.json")

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
VECTOR_TOP_K = 4
GRAPH_MAX_HOPS = 2