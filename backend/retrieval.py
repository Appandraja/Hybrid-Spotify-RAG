import json
import os
from typing import List, Dict, Any, Tuple
import networkx as nx
from networkx.readwrite import json_graph
import chromadb
from openai import OpenAI

from backend.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    GRAPH_STORE_PATH,
    VECTOR_TOP_K
)

client = OpenAI(api_key=OPENAI_API_KEY)

def load_graph() -> nx.DiGraph:
    if not os.path.exists(GRAPH_STORE_PATH):
        return nx.DiGraph()
    with open(GRAPH_STORE_PATH, "r") as f:
        data = json.load(f)
    return json_graph.node_link_graph(data)

def vector_search(query: str, top_k: int = VECTOR_TOP_K) -> List[Dict[str, Any]]:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    try:
        collection = chroma_client.get_collection(name="pdf_documents")
    except Exception:
        return []

    query_emb = client.embeddings.create(
        input=[query],
        model=EMBEDDING_MODEL
    ).data[0].embedding

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k
    )

    extracted = []
    if results and results.get("documents"):
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        ids = results["ids"][0]
        distances = results.get("distances", [[]])[0]

        for idx in range(len(docs)):
            extracted.append({
                "id": ids[idx],
                "text": docs[idx],
                "page": metas[idx].get("page", 1),
                "score": round(1 - distances[idx], 4) if distances else 0.0
            })
    return extracted

def graph_search(query: str) -> List[Dict[str, Any]]:
    graph = load_graph()
    if graph.number_of_nodes() == 0:
        return []

    matched_edges = []
    query_words = set(query.lower().split())

    # Find nodes matching terms in query
    matching_nodes = [
        node for node in graph.nodes()
        if any(word in node.lower() for word in query_words if len(word) > 2)
    ]

    for node in matching_nodes:
        # Collect 1-hop outgoing and incoming edges
        for u, v, data in graph.out_edges(node, data=True):
            matched_edges.append({
                "source": u,
                "target": v,
                "relation": data.get("relation", "RELATED_TO"),
                "page": data.get("page", "N/A")
            })
        for u, v, data in graph.in_edges(node, data=True):
            matched_edges.append({
                "source": u,
                "target": v,
                "relation": data.get("relation", "RELATED_TO"),
                "page": data.get("page", "N/A")
            })

    # Deduplicate matching edge triples
    unique_edges = []
    seen = set()
    for e in matched_edges:
        key = (e["source"], e["relation"], e["target"])
        if key not in seen:
            seen.add(key)
            unique_edges.append(e)

    return unique_edges[:10]