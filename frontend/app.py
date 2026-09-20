import streamlit as st
import requests
import json
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Graph-Vector RAG Studio",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Spotify WebApplication")
st.caption("Powered by FastAPI, ChromaDB, NetworkX, PyPDF, and OpenAI")

# Sidebar Configuration and Upload
with st.sidebar:
    st.header("📄 Document Ingestion")
    uploaded_file = st.file_uploader("Upload architecture PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Process & Ingest PDF", type="primary"):
            with st.spinner("Extracting text, embedding chunks into ChromaDB, and building NetworkX Graph..."):
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf"
                    )
                }

                try:
                    res = requests.post(
                        f"{BACKEND_URL}/upload",
                        files=files
                    )

                    if res.status_code == 200:
                        data = res.json()
                        st.success("PDF successfully processed!")
                        st.json(data)
                    else:
                        st.error(f"Error: {res.text}")

                except Exception as e:
                    st.error(f"Connection failed: {e}")

    st.divider()
    st.header("⚙️ Settings")

    api_url = BACKEND_URL

# Main UI Tabs
tab_chat, tab_graph, tab_evals = st.tabs(["💬 Chat & Search", "🕸️ Knowledge Graph", "🧪 Automated Evals"])

# TAB 1: Chat Assistant & Evidence Inspector
with tab_chat:
    st.subheader("Ask questions about your uploaded PDF")
    
    query = st.text_input("Enter your query:", placeholder="e.g., Which microservices manage user subscriptions and payment processing?")
    top_k = st.slider("Vector Chunks (Top K)", min_value=1, max_value=10, value=4)

    if st.button("Generate Answer", type="primary"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving vector & graph evidence and synthesizing response..."):
                try:
                    response = requests.post(
                        f"{api_url}/query",
                        json={"query": query, "top_k": top_k}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        st.markdown("### Answer")
                        st.info(data["answer"])
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 📚 Vector Evidence (ChromaDB)")
                            for i, chunk in enumerate(data["vector_evidence"]):
                                with st.expander(f"Chunk {i+1} - Page {chunk['page']} (Score: {chunk['score']})"):
                                    st.write(chunk["text"])

                        with col2:
                            st.markdown("#### 🕸️ Graph Evidence (NetworkX)")
                            if data["graph_evidence"]:
                                for edge in data["graph_evidence"]:
                                    st.markdown(f"- **{edge['source']}** `--[{edge['relation']}]-->` **{edge['target']}**")
                            else:
                                st.write("No direct graph relationships extracted for this query.")
                    else:
                        st.error(f"Error ({response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")

# TAB 2: Knowledge Graph Explorer
with tab_graph:
    st.subheader("NetworkX Knowledge Graph Inspection")
    if st.button("Refresh Graph Stats"):
        try:
            res = requests.get(f"{api_url}/graph-info")
            if res.status_code == 200:
                stats = res.json()
                c1, c2 = st.columns(2)
                c1.metric("Total Extracted Entities (Nodes)", stats["num_nodes"])
                c2.metric("Total Entity Relations (Edges)", stats["num_edges"])
                
                st.markdown("#### Sample Nodes")
                st.json(stats["sample_nodes"])
                
                st.markdown("#### Sample Edge Triples")
                st.json(stats["sample_edges"])
            else:
                st.error("Failed to load graph info.")
        except Exception as e:
            st.error(f"Error fetching graph stats: {e}")

# TAB 3: Automated Functional Evaluations
with tab_evals:
    st.subheader("Run Functional Suite")
    st.write("Executes automated checks for prompt injection guardrails, vector retrieval accuracy, and knowledge graph grounding.")
    
    if st.button("Run Evaluations"):
        with st.spinner("Executing evaluation benchmarks..."):
            try:
                res = requests.post(f"{api_url}/evaluate", json={})
                if res.status_code == 200:
                    eval_data = res.json()
                    st.success("Evaluations Completed!")
                    
                    st.markdown("### Summary Metrics")
                    st.json(eval_data["summary"])
                    
                    st.markdown("### Individual Query Results")
                    st.table(eval_data["query_evaluations"])
                else:
                    st.error("Evaluation execution failed.")
            except Exception as e:
                st.error(f"Error calling evaluation endpoint: {e}")