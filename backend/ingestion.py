import io
import json
import os
from typing import List, Dict, Any
from pypdf import PdfReader
import networkx as nx
from networkx.readwrite import json_graph
import chromadb
from openai import OpenAI

from backend.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    LLM_MODEL,
    CHROMA_PERSIST_DIR,
    GRAPH_STORE_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

client = OpenAI(api_key=OPENAI_API_KEY)

def extract_text_from_pdf(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    documents = []
    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            documents.append({
                "page": page_idx + 1,
                "text": text.strip()
            })
    return documents

def chunk_text(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks = []
    chunk_id = 0
    for doc in documents:
        words = doc["text"].split()
        page_num = doc["page"]
        
        i = 0
        while i < len(words):
            chunk_words = words[i:i + CHUNK_SIZE]
            chunk_text_str = " ".join(chunk_words)
            chunks.append({
                "chunk_id": f"chunk_{chunk_id}",
                "page": page_num,
                "text": chunk_text_str
            })
            chunk_id += 1
            i += (CHUNK_SIZE - CHUNK_OVERLAP) if len(words) > CHUNK_SIZE else len(words)
            if i >= len(words):
                break
    return chunks

def store_in_chroma(chunks: List[Dict[str, Any]]):
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    
    try:
        chroma_client.delete_collection("pdf_documents")
    except Exception:
        pass

    collection = chroma_client.create_collection(name="pdf_documents")

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [{"page": c["page"]} for c in chunks]

    # Generate embeddings via OpenAI
    response = client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL
    )
    embeddings = [res.embedding for res in response.data]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

def extract_graph_relations(chunks: List[Dict[str, Any]]) -> nx.DiGraph:
    graph = nx.DiGraph()
    
    prompt_template = """You are an expert knowledge graph extractor.
Analyze the following document chunk and extract key entities and their semantic relationships.

Return ONLY a valid JSON object matching this schema:
{
  "triples": [
    {
      "subject": "Entity A",
      "subject_type": "Service/Database/Tech/Feature",
      "relation": "USES / STORES / DEPENDS_ON / MANAGES / POWERED_BY",
      "object": "Entity B",
      "object_type": "Service/Database/Tech/Feature"
    }
  ]
}

Text Chunk:
\"\"\"
{chunk_text}
\"\"\"
"""

    # Sample representative chunks if doc is large to ensure speed and budget efficiency
    sampled_chunks = chunks[:15] if len(chunks) > 15 else chunks

    for chunk in sampled_chunks:
        try:
            res = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You extract structured knowledge graph triples from technical documentation."},
                    {"role": "user", "content": prompt_template.format(chunk_text=chunk["text"])}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            content = res.choices[0].message.content
            data = json.loads(content)
            
            for item in data.get("triples", []):
                subj = item.get("subject", "").strip()
                obj = item.get("object", "").strip()
                rel = item.get("relation", "RELATED_TO").strip()
                
                if subj and obj:
                    graph.add_node(subj, type=item.get("subject_type", "Concept"))
                    graph.add_node(obj, type=item.get("object_type", "Concept"))
                    graph.add_edge(subj, obj, relation=rel, page=chunk["page"])
        except Exception as e:
            print(f"Error extracting graph for chunk {chunk['chunk_id']}: {e}")
            continue

    # Save NetworkX graph to disk
    os.makedirs(os.path.dirname(GRAPH_STORE_PATH), exist_ok=True)
    graph_data = json_graph.node_link_data(graph)
    with open(GRAPH_STORE_PATH, "w") as f:
        json.dump(graph_data, f, indent=2)

    return graph

def process_and_ingest_pdf(pdf_bytes: bytes) -> Dict[str, Any]:
    docs = extract_text_from_pdf(pdf_bytes)
    chunks = chunk_text(docs)
    store_in_chroma(chunks)
    graph = extract_graph_relations(chunks)
    
    return {
        "status": "success",
        "total_pages": len(docs),
        "total_chunks": len(chunks),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges()
    }