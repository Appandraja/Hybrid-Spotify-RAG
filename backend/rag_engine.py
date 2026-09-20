from typing import Dict, Any
from openai import OpenAI
from backend.config import OPENAI_API_KEY, LLM_MODEL
from backend.retrieval import vector_search, graph_search

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_answer(query: str, top_k: int = 4) -> Dict[str, Any]:
    vector_results = vector_search(query, top_k=top_k)
    graph_results = graph_search(query)

    # Build context string
    vector_context = "\n\n".join([
        f"[Source Chunk (Page {res['page']})]:\n{res['text']}" 
        for res in vector_results
    ])

    graph_context = "\n".join([
        f"- ({edge['source']}) --[{edge['relation']}]--> ({edge['target']})" 
        for edge in graph_results
    ])

    system_prompt = """You are an intelligent technical assistant answering questions based on retrieved documentation and knowledge graph relations.

Guidelines:
1. Ground your answer strictly in the provided Vector Text Chunks and Knowledge Graph Relations.
2. Explicitly reference source pages or relations where applicable.
3. Be direct, clear, and precise. If the information is missing from the context, state that clearly."""

    user_content = f"""Query: {query}

--- RETRIEVED TEXT CHUNKS (VECTOR SEARCH) ---
{vector_context if vector_context else "No relevant text chunks found."}

--- RETRIEVED KNOWLEDGE GRAPH RELATIONS ---
{graph_context if graph_context else "No direct entity graph relations found."}

Please answer the user's question clearly based on the context above."""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        temperature=0.2
    )

    return {
        "query": query,
        "answer": response.choices[0].message.content,
        "vector_evidence": vector_results,
        "graph_evidence": graph_results
    }