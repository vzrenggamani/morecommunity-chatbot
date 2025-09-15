import streamlit as st
import os
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# --- App Configuration ---
st.set_page_config(page_title="Chatbot Dukungan Orang Tua", page_icon="❤️")
st.title("❤️ Chatbot Dukungan untuk Orang Tua Hebat")
st.write("Dapatkan informasi dari komunitas dan para ahli. Ketik pertanyaan Anda di bawah ini.")

# --- Google API Key Setup ---
# For Streamlit Cloud, use st.secrets to keep your API key secure
try:
    # Local development: Use an environment variable
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        # Deployed on Streamlit Cloud: Use secrets
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
except (KeyError, TypeError):
    st.error("Error: GOOGLE_API_KEY not found. Please set it in Streamlit secrets or as an environment variable.")
    st.stop()


# --- Caching Functions to Load Data and Models Only Once ---
@st.cache_resource
def load_llm_and_retriever():
    """
    Loads all necessary components: documents, vector store, and the QA chain.
    The @st.cache_resource decorator ensures this function only runs once.
    """
    # 1. Load Documents from the 'data' directory
    loader = DirectoryLoader('./data/', glob="**/*.md", loader_cls=UnstructuredMarkdownLoader, show_progress=True)
    documents = loader.load()

    # 2. Split Documents into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # 3. Create Embeddings and Store in ChromaDB
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # Persist the vector store in a local directory
    vector_store = Chroma.from_documents(texts, embeddings, persist_directory="chroma_db")

    # 4. Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, convert_system_message_to_human=True)

    # 5. Create the RetrievalQA chain
    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 most relevant chunks
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True # This is useful for showing sources
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
