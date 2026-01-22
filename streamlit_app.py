import streamlit as st
import sys
import os
import json
import rdflib

# Add 'app' directory to path to import reasoning_engine
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# [Cloud Fix] Inject Secrets to Environment for reasoning_engine.py
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except FileNotFoundError:
    pass
except Exception as e:
    print(f"Warning: Issue accessing st.secrets: {e}")

if "GOOGLE_API_KEY" not in os.environ and "GOOGLE_API_KEY" not in st.secrets:
    pass

try:
    from reasoning_engine import generate_sparql, execute_sparql, generate_answer, DEFAULT_SPARQL_PROMPT, DEFAULT_ANSWER_PROMPT
except ValueError as e:
    st.error("🚨 **Deployment Error: Google API Key Missing**")
    st.warning("Please configure your Secrets in Streamlit Cloud Settings.")
    st.stop()

from graph_loader import load_graph, generate_schema_info
from visualize_graph import visualize_ontology

# Page Config
st.set_page_config(page_title="Math Ontology Prompt Playground", layout="wide")

# Paths
TBOX_PATH = "data/ontology/math_tbox.ttl"
DATA_PATH = "data/knowledge_graph/math_abox.ttl"
VISUALIZATION_PATH = "math_graph.html"

# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "graph_loaded" not in st.session_state:
    st.session_state.graph_loaded = False
if "viz_html" not in st.session_state:
    st.session_state.viz_html = None

# PROMPT MANAGEMENT (NEW)
st.title("🛠️ Prompt Engineering App")

with st.expander("⚙️  Configure Prompts", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("SPARQL Query Prompt")
        sparql_prompt_template = st.text_area(
            "Instructions for converting Question to SPARQL:", 
            value=DEFAULT_SPARQL_PROMPT, 
            height=300,
            key="sparql_prompt"
        )
    with col2:
        st.subheader("Answer Generation Prompt")
        answer_prompt_template = st.text_area(
            "Instructions for generating final Answer:", 
            value=DEFAULT_ANSWER_PROMPT, 
            height=300,
            key="answer_prompt"
        )

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🗺️ Ontology Map")
    
    # Simple Reset Button
    if st.button("Reset View"):
        st.session_state.viz_html = None # Reset visualization
        st.rerun()

    # Visualization
    if st.session_state.viz_html:
        st.components.v1.html(st.session_state.viz_html, height=700, scrolling=True)
    elif os.path.exists(VISUALIZATION_PATH):
        with open(VISUALIZATION_PATH, 'r', encoding='utf-8') as f:
            html_data = f.read()
        st.components.v1.html(html_data, height=700, scrolling=True)
    else:
        st.info("Visualization loading...")

# Main Load Logic
@st.cache_resource
def get_graph_data():
    g = load_graph(DATA_PATH)
    t = load_graph(TBOX_PATH)
    full_g = g + t
    schema = generate_schema_info(full_g)
    return full_g, schema

try:
    full_graph, schema_info = get_graph_data()
    st.session_state.graph_loaded = True
except Exception as e:
    st.error(f"Failed to load graph: {e}")
    st.stop()

# Title
st.subheader("💬 Chat Interface")

# Chat UI - Main Area
# Chat History
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "evidence" in msg and msg["evidence"]:
            with st.expander("🔍 Evidence"):
                st.table(msg["evidence"])

# Chat Input (Pinned to bottom)
if prompt := st.chat_input("Ask a math question..."):
    # User Message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.spinner("Analyzing Ontology with YOUR prompts..."):
        # 1. Reasoning (Pass Custom Prompt)
        sparql_res = generate_sparql(prompt, schema_info, prompt_template=sparql_prompt_template)
        
        # 2. Execution
        db_data = []
        if sparql_res and "query" in sparql_res and sparql_res["query"]:
             db_data = execute_sparql(sparql_res["query"], full_graph)
        
        # 3. Answer Generation (Pass Custom Prompt)
        final_res = generate_answer(prompt, db_data, sparql_res.get("explanation", ""), prompt_template=answer_prompt_template)
        
        answer_text = final_res.get("answer", "No answer generated.")
        evidence_data = final_res.get("evidence", [])
        
        # 4. Update Visualization (Highlighting)
        if evidence_data:
            highlight_nodes = []
            for item in evidence_data:
                if item.get("concept"): highlight_nodes.append(item["concept"])
                if item.get("chapter"): highlight_nodes.append(item["chapter"])
                if item.get("subject"): highlight_nodes.append(item["subject"])
            
            try:
                new_html = visualize_ontology(graph=full_graph, highlight_labels=highlight_nodes, return_html_str=True)
                st.session_state.viz_html = new_html
            except Exception as e:
                print(f"Visualization Error: {e}")

    # Assistant Message
    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": answer_text,
        "evidence": evidence_data
    })
    
    st.rerun()
