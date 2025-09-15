# Patch for ChromaDB compatibility on Streamlit Cloud
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import glob #<-- Import glob

# ... (keep your streamlit config and API key setup the same) ...

@st.cache_resource
def load_llm_and_retriever():
    """
    Loads all necessary components and handles errors gracefully.
    """
    # 1. Load Documents with improved error handling
    st.info("Loading documents from the 'data' directory...")
    all_docs = []
    # Find all markdown files in the data directory
    md_files = glob.glob('./data/**/*.md', recursive=True)
    
    for file_path in md_files:
        try:
            loader = UnstructuredMarkdownLoader(file_path)
            # .load() returns a list, so we extend our list
            all_docs.extend(loader.load())
        except Exception as e:
            # If a file fails, print a warning and skip it
            st.warning(f"Could not load file: {file_path}. Error: {e}")
            continue # Go to the next file
    
    if not all_docs:
        st.error("No documents were loaded. Please check the 'data' folder and file contents.")
        st.stop()
        
    st.success(f"Successfully loaded {len(all_docs)} document(s).")

    # 2. Split Documents into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(all_docs)

    # 3. Create Embeddings and Store in ChromaDB
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = Chroma.from_documents(texts, embeddings, persist_directory="chroma_db")

    # 4. Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, convert_system_message_to_human=True)

    # 5. Create the RetrievalQA chain
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# --- Main Application Logic ---
# Load the QA chain (this will be cached after the first run)
try:
    qa_chain = load_llm_and_retriever()
except Exception as e:
    st.error(f"Failed to initialize the chatbot. Error: {e}")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Apa yang ingin Anda tanyakan?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Mencari jawaban terbaik untuk Anda..."):
        # Get the response from the RAG chain
        response = qa_chain.invoke(prompt)
        answer = response['result']
        
        # Optionally, display the sources for transparency
        sources = response['source_documents']
        source_list = "\n\n**Sumber Informasi:**\n"
        for i, doc in enumerate(sources):
            # Assumes the source path is useful, e.g., 'data/nama_file.md'
            source_list += f"- {os.path.basename(doc.metadata.get('source', 'Unknown'))}\n"

        full_response = answer + source_list

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
