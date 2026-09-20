# Hybrid-Spotify-RAG
Hybrid Graph-Vector RAG Application An enterprise-ready Retrieval-Augmented Generation (RAG) system that combines traditional Vector Search (ChromaDB) with Knowledge Graph Retrieval (NetworkX) to provide precise, grounded, and context-aware answers from PDF documentation.

The system is equipped with FastAPI backend microservices, an interactive Streamlit UI, built-in prompt-injection guardrails, and an automated evaluation suite.
📐 System Architecture
                      +-------------------+
                      |   PDF Document    |
                      +---------+---------+
                                |
                                v
                   Text Extraction & Chunking
                                |
             +------------------+------------------+
             |                                     |
             v                                     v
     OpenAI Embeddings                 Entity & Relation Extraction
             |                                     |
             v                                     v
   ChromaDB (Vector DB)                  NetworkX (Graph Store)
             |                                     |
             +------------------+------------------+
                                |
                                v
                          User Question
                                |
             +------------------+------------------+
             |                                     |
             v                                     v
       Vector Search                         Graph Search
    (Semantic Similarity)                 (Entity Context)
             |                                     |
             +------------------+------------------+
                                |
                                v
                        Combined Evidence
                                |
                                v
                           OpenAI LLM
                        (Answer Generation)


✨ Features
Hybrid Retrieval System: Combines vector similarity search using OpenAI embeddings with entity-relationship search using NetworkX.
FastAPI Backend: Fully asynchronous backend API with response models, validation, and auto-generated OpenAPI (Swagger) documentation.
Streamlit Frontend: Clean user interface to upload PDFs, run questions, visualize vector and graph evidence, and execute tests.
Security Guardrails: Basic validation and regex-based prompt-injection filters using Pydantic.
Automated Evaluations: Built-in evaluation routines to test guardrail effectiveness and document grounding score.
Containerized Deployment: Multi-stage Docker builds with Docker Compose for production deployments.
🛠️ Tech Stack
Language: Python 3.11
Backend: FastAPI, Uvicorn
Frontend: Streamlit
Vector Database: ChromaDB
Knowledge Graph: NetworkX
LLM & Embeddings: OpenAI (gpt-4o-mini, text-embedding-3-small)
PDF Processing: PyPDF
Validation: Pydantic v2
Containerization: Docker, Docker Compose
📂 Project Structure
.
├── Dockerfile.backend      # Docker build configuration for FastAPI service
├── Dockerfile.frontend     # Docker build configuration for Streamlit UI
├── docker-compose.yml      # Orchestration for containerized deployment
├── requirements.txt        # Python dependency requirements
├── .env                    # Environment variables (OpenAI API key)
├── backend/
│   ├── config.py           # Application configurations & env vars
│   ├── evals.py            # Automated evaluation routines
│   ├── guardrails.py       # Pydantic schemas and prompt-injection checks
│   ├── ingestion.py        # PDF text extraction, chunking, graph generation
│   ├── main.py             # FastAPI entry point and routes
│   ├── rag_engine.py       # Context assembly and LLM answer generation
│   └── retrieval.py        # Vector and graph search functions
└── frontend/
    └── app.py              # Streamlit web interface


🚀 Quick Start (Local Development)
1. Prerequisites
Python 3.11+
An OpenAI API Key
2. Installation
Clone the repository and set up a virtual environment:
git clone https://github.com/your-username/hybrid-rag-app.git
cd hybrid-rag-app

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


3. Environment Setup
Create a .env file in the root directory:
OPENAI_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini


4. Run the Backend (FastAPI)
python -m backend.main


The API will start at http://localhost:8000. You can inspect the Swagger documentation at http://localhost:8000/docs.
5. Run the Frontend (Streamlit)
In a new terminal window (with the virtual environment activated):
streamlit run frontend/app.py


Access the application UI at http://localhost:8501.
🐳 Running with Docker Compose (Production)
To deploy both the backend and frontend services using Docker:
1. Configure .env
Ensure your .env file contains your OpenAI key.
2. Build and Start Containers
docker-compose up --build -d


3. Verify Deployment
Streamlit Web Interface: http://localhost:8501
FastAPI Documentation: http://localhost:8000/docs
4. Stop Services
docker-compose down


📊 API Endpoints
Method
Endpoint
Description
GET
/
Health check endpoint
POST
/upload
Upload and process a PDF document
POST
/query
Query the hybrid RAG system
GET
/graph-info
Fetch Knowledge Graph metadata and node samples
POST
/evaluate
Run automated functional benchmarks

🧪 Running Functional Evaluations
You can execute functional evaluations directly from the Automated Evals tab in the Streamlit UI, or programmatically by sending a POST request to the /evaluate API endpoint:
curl -X 'POST' \
  'http://localhost:8000/evaluate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "test_queries": [
    "What microservices manage user profiles and auth?",
    "What database is used for analytics and audio events?"
  ]
}'


🛡️ Guardrails & Safety
The application implements input validation via Pydantic models in backend/guardrails.py. Queries are inspected for common prompt-injection attempts (e.g., ignore previous instructions, jailbreak, etc.) and rejected before reaching vector storage or the LLM.

