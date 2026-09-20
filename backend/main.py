from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from backend.guardrails import QueryRequest, QueryResponse, EvaluationRequest
from backend.ingestion import process_and_ingest_pdf
from backend.rag_engine import generate_answer
from backend.evals import run_functional_evaluations
from backend.retrieval import load_graph

app = FastAPI(
    title="Hybrid Graph-Vector RAG API",
    description="FastAPI service for PDF processing, ChromaDB vector retrieval, NetworkX Knowledge Graph, and LLM orchestration.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hybrid Graph-Vector RAG Service is running."}

@app.post("/upload", summary="Ingest PDF document")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    contents = await file.read()
    res = process_and_ingest_pdf(contents)
    return res

@app.post("/query", response_model=QueryResponse, summary="Query RAG System")
def query_rag(req: QueryRequest):
    result = generate_answer(query=req.query, top_k=req.top_k)
    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        vector_evidence=result["vector_evidence"],
        graph_evidence=result["graph_evidence"],
        guardrail_passed=True
    )

@app.get("/graph-info", summary="Get Knowledge Graph Stats")
def get_graph_info():
    graph = load_graph()
    nodes = list(graph.nodes(data=True))[:20]
    edges = list(graph.edges(data=True))[:20]
    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "sample_nodes": [{"id": n[0], "attributes": n[1]} for n in nodes],
        "sample_edges": [{"source": e[0], "target": e[1], "attributes": e[2]} for e in edges]
    }

@app.post("/evaluate", summary="Run Functional Evaluations")
def run_evals(req: EvaluationRequest):
    return run_functional_evaluations(req.test_queries)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)