# Patch for ChromaDB compatibility on Streamlit Cloud
__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
from pages.chat_page import show_chat_page
from pages.debug_page import show_debug_page
from pages.raw_response_page import show_raw_response_page
from pages.direct_llm_page import show_direct_llm_page
from pages.experiment_page import show_qa_vs_llm_experiment_page
from utils.llm_utils import load_llm_and_retriever

# --- Main Application Logic ---
st.set_page_config(
    page_title="Rare Disease Helper Chatbot", page_icon="🏥", layout="wide"
)

st.sidebar.title("🏥 Menu")
st.sidebar.markdown("Navigate through the chatbot features")

# Navigation
page = st.sidebar.selectbox(
    "Select Page",
    ["💬 Chat", "🔧 Debug Info", "🔍 Raw Response", "🤖 Direct LLM", "🧪 Experiment"],
    index=0,
)

# Add logo and title
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("# 🏥 Rare Disease Helper Chatbot")

st.markdown("---")  # Add a separator line

# Load the QA chain (this will be cached after the first run)
try:
    qa_chain = load_llm_and_retriever()
except Exception as e:
    st.error(f"Gagal untuk memuat layanan. Error: {e}")
    st.stop()

# Page routing
if page == "🔧 Debug Info":
    show_debug_page()
elif page == "🔍 Raw Response":
    show_raw_response_page(qa_chain)
elif page == "🤖 Direct LLM":
    show_direct_llm_page()  # This page doesn't need qa_chain
elif page == "🧪 Experiment":
    show_qa_vs_llm_experiment_page()  # Comparative analysis page
elif page == "💬 Chat":
    show_chat_page(qa_chain)
