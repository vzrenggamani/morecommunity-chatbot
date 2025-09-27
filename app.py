# Patch for ChromaDB compatibility on Streamlit Cloud
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import sys
from datetime import datetime
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import glob #<-- Import glob

# ... (keep your streamlit config and API key setup the same) ...

# --- Utility Functions ---
def determine_document_type(file_path):
    """Determine document type based on folder structure"""
    if 'medical_reference' in file_path:
        return 'medical_reference'
    elif 'user_stories' in file_path:
        return 'user_story'
    elif 'community_resources' in file_path:
        return 'community_resource'
    elif 'clinical_guidelines' in file_path:
        return 'clinical_guideline'
    else:
        return 'general'

def should_rebuild_vectorstore(vector_store_path, data_path='./data'):
    """Check if vector store needs rebuilding based on file timestamps"""
    if not os.path.exists(vector_store_path):
        return True, "Vector store tidak ditemukan"

    # Get vector store timestamp
    try:
        vectorstore_timestamp = os.path.getmtime(vector_store_path)
    except:
        return True, "Tidak dapat membaca timestamp vector store"

    # Check all markdown files
    md_files = glob.glob(f'{data_path}/**/*.md', recursive=True)
    if not md_files:
        return False, "Tidak ada file markdown ditemukan"

    # Check if any document is newer than vector store
    for file_path in md_files:
        try:
            file_timestamp = os.path.getmtime(file_path)
            if file_timestamp > vectorstore_timestamp:
                return True, f"Dokumen {os.path.basename(file_path)} telah diperbarui"
        except:
            continue

    return False, "Vector store masih up-to-date"

@st.cache_resource
def load_llm_and_retriever():
    """
    Loads components with persistent vector store to minimize cold starts.
    """

    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    # Smart caching: Check if vector store needs rebuilding
    vector_store_path = "chroma_db"
    needs_rebuild, reason = should_rebuild_vectorstore(vector_store_path)

    if not needs_rebuild:
        st.info("📚 Memuat vector store yang sudah ada...")
        try:
            # Load existing vector store
            vector_store = Chroma(
                persist_directory=vector_store_path,
                embedding_function=embeddings
            )
            st.success(f"✅ Vector store berhasil dimuat dari cache! ({reason})")
        except Exception as e:
            st.warning(f"⚠️ Gagal memuat vector store cache: {e}")
            needs_rebuild = True
            reason = "Cache corrupted"

    # Rebuild if needed
    if needs_rebuild:
        st.info(f"🔄 Membangun ulang vector store: {reason}")

        # 1. Load Documents with type labeling
        st.info("📖 Memuat dokumen dari direktori 'data'...")
        all_docs = []
        md_files = glob.glob('./data/**/*.md', recursive=True)

        for file_path in md_files:
            try:
                loader = UnstructuredMarkdownLoader(file_path)
                docs = loader.load()

                # Add document type metadata
                doc_type = determine_document_type(file_path)
                for doc in docs:
                    doc.metadata['document_type'] = doc_type
                    doc.metadata['source_category'] = doc_type
                    # Keep original source path
                    doc.metadata['original_source'] = file_path

                all_docs.extend(docs)
                st.info(f"✅ Memuat {len(docs)} dokumen dari {doc_type}: {os.path.basename(file_path)}")

            except Exception as e:
                # If a file fails, print a warning and skip it
                st.warning(f"❌ Tidak dapat memuat file: {file_path}. Error: {e}")
                continue # Go to the next file

        if not all_docs:
            st.error("Tidak ada dokumen yang dimuat. Silakan periksa folder 'data' dan isi file.")
            st.stop()

        # Show summary of loaded documents by type
        doc_types = {}
        for doc in all_docs:
            doc_type = doc.metadata.get('document_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

        summary = ", ".join([f"{count} {dtype}" for dtype, count in doc_types.items()])
        st.success(f"✅ Berhasil memuat {len(all_docs)} dokumen total: {summary}")

        # 2. Split Documents into Chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(all_docs)

        # 3. Create Embeddings and Store in ChromaDB
        st.info("🔄 Membuat embeddings dan menyimpan ke vector store...")
        vector_store = Chroma.from_documents(texts, embeddings, persist_directory=vector_store_path)
        st.success("✅ Vector store baru berhasil dibuat dan disimpan!")

    # 4. Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3, convert_system_message_to_human=True)

    # 5. Create custom prompt template that handles different document types
    custom_prompt_template = """
    Anda adalah dokter AI yang ahli dalam bidang penyakit langka dan kesehatan. Anda memiliki akses ke berbagai jenis pengetahuan medis:

    SUMBER PENGETAHUAN ANDA:
    {context}

    Ketika memberikan jawaban, sesuaikan pendekatan berdasarkan jenis informasi:
    - Informasi medis/klinis: Gunakan dengan otoritas profesional dan presisi klinis
    - Pengalaman pasien: Gunakan untuk memberikan empati dan pemahaman mendalam tentang pengalaman keluarga
    - Sumber komunitas: Gunakan untuk memberikan dukungan dan informasi praktis tentang resources

    PERTANYAAN PASIEN: {question}

    PENTING - JANGAN LAKUKAN:
    - JANGAN katakan "berdasarkan cerita Anda", "dari informasi yang Anda berikan"
    - JANGAN asumsikan pasien telah menceritakan detail yang ada di pengetahuan Anda
    - JANGAN rujuk ke sumber atau dokumen secara eksplisit

    LAKUKAN:
    - Jawab langsung dan profesional seperti dokter berpengalaman
    - Integrasikan semua jenis pengetahuan secara natural
    - Berikan empati berdasarkan pengalaman keluarga yang Anda ketahui
    - Sertakan informasi praktis dan dukungan komunitas jika relevan
    - Selalu sarankan konsultasi medis profesional untuk diagnosis dan pengobatan

    JAWABAN DOKTER:"""

    PROMPT = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

    # 6. Create the RetrievalQA chain with custom prompt
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 5,
            "score_threshold": 0.3  # Lower threshold = more lenient matching
        }
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# --- Page Functions ---
def show_debug_page():
    """Display debug information about the vector store and system"""
    st.title("🔧 Debug Information")
    st.markdown("This page shows technical information about the system for debugging purposes.")

    # System Information
    st.header("📊 System Information")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Environment")
        st.write(f"**Python Version:** {sys.version.split()[0]}")
        st.write(f"**Current Working Directory:** {os.getcwd()}")
        st.write(f"**Google API Key Set:** {'✅ Yes' if os.getenv('GOOGLE_API_KEY') else '❌ No'}")

    with col2:
        st.subheader("Directories")
        data_dir = "./data"
        vector_store_dir = "chroma_db"
        st.write(f"**Data Directory Exists:** {'✅ Yes' if os.path.exists(data_dir) else '❌ No'}")
        st.write(f"**Vector Store Exists:** {'✅ Yes' if os.path.exists(vector_store_dir) else '❌ No'}")

    # Document Information
    st.header("📄 Document Information")

    # Check markdown files
    md_files = glob.glob('./data/**/*.md', recursive=True)
    if md_files:
        st.success(f"Found {len(md_files)} markdown files")

        # Group by document type
        doc_types = {}
        for file_path in md_files:
            doc_type = determine_document_type(file_path)
            if doc_type not in doc_types:
                doc_types[doc_type] = []
            doc_types[doc_type].append(file_path)

        # Display by type
        for doc_type, files in doc_types.items():
            with st.expander(f"📁 {doc_type.replace('_', ' ').title()} ({len(files)} files)"):
                for file_path in files:
                    file_name = os.path.basename(file_path)
                    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                    file_modified = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S") if os.path.exists(file_path) else "Unknown"

                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**{file_name}**")
                    with col2:
                        st.write(f"{file_size} bytes")
                    with col3:
                        st.write(file_modified)
    else:
        st.warning("No markdown files found in data directory")

    # Vector Store Information
    st.header("🗄️ Vector Store Information")

    vector_store_path = "chroma_db"

    # Check both if directory exists and if it contains valid ChromaDB files
    vector_store_exists = os.path.exists(vector_store_path)
    has_chroma_files = False

    if vector_store_exists:
        # Check for ChromaDB specific files
        chroma_files = ['chroma.sqlite3', 'chroma.sqlite3-wal', 'chroma.sqlite3-shm']
        has_chroma_files = any(os.path.exists(os.path.join(vector_store_path, f)) for f in chroma_files)

    if vector_store_exists:
        if has_chroma_files:
            st.success("Vector store directory exists with ChromaDB files")
        else:
            st.warning("Vector store directory exists but appears empty or invalid")

        try:
            # Load vector store to get information
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            vector_store = Chroma(
                persist_directory=vector_store_path,
                embedding_function=embeddings
            )

            # Get collection information
            collection = vector_store._collection
            count = collection.count()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Documents", count)

                # Calculate vector store size more safely
                try:
                    vector_store_size = 0
                    for root, dirs, files in os.walk(vector_store_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            if os.path.isfile(file_path):
                                vector_store_size += os.path.getsize(file_path)
                    st.metric("Vector Store Size", f"{vector_store_size / 1024:.1f} KB")
                except Exception as e:
                    st.error(f"Error calculating size: {e}")

            with col2:
                try:
                    vector_store_modified = datetime.fromtimestamp(os.path.getmtime(vector_store_path)).strftime("%Y-%m-%d %H:%M:%S")
                    st.write(f"**Last Modified:** {vector_store_modified}")
                except Exception as e:
                    st.write(f"**Last Modified:** Error - {e}")

                # Check if rebuild is needed
                needs_rebuild, reason = should_rebuild_vectorstore(vector_store_path)
                status = "🔄 Needs Rebuild" if needs_rebuild else "✅ Up to Date"
                st.write(f"**Status:** {status}")
                st.write(f"**Reason:** {reason}")

                # Show if vector store is actually empty
                if count == 0:
                    st.error("⚠️ Vector store is empty! Documents not loaded.")

            # Additional ChromaDB debugging info
            with st.expander("🔧 ChromaDB Technical Details"):
                try:
                    st.write(f"**Collection Name:** {collection.name}")
                    st.write(f"**Collection ID:** {collection.id}")

                    # List files in vector store directory
                    st.write("**Files in vector store directory:**")
                    for root, dirs, files in os.walk(vector_store_path):
                        level = root.replace(vector_store_path, '').count(os.sep)
                        indent = ' ' * 2 * level
                        st.code(f"{indent}{os.path.basename(root)}/")
                        subindent = ' ' * 2 * (level + 1)
                        for file in files:
                            file_path = os.path.join(root, file)
                            size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                            st.code(f"{subindent}{file} ({size} bytes)")

                except Exception as e:
                    st.error(f"Error getting ChromaDB details: {e}")

            # Sample documents
            if st.button("Show Sample Documents"):
                if count > 0:
                    try:
                        # Get a few sample documents
                        results = collection.peek(limit=3)

                        st.subheader("Sample Document Chunks")
                        for i, (doc_id, metadata, document) in enumerate(zip(results['ids'], results['metadatas'], results['documents'])):
                            with st.expander(f"Document {i+1} - {doc_id[:50]}..."):
                                st.write("**Metadata:**")
                                st.json(metadata)
                                st.write("**Content Preview:**")
                                st.write(document[:500] + "..." if len(document) > 500 else document)

                    except Exception as e:
                        st.error(f"Error retrieving sample documents: {e}")
                else:
                    st.warning("No documents to show - vector store is empty")

        except Exception as e:
            st.error(f"Error loading vector store: {e}")
            st.info("This usually means the vector store is corrupted or was not properly created.")
    else:
        st.error("Vector store directory does not exist")
        st.info("The vector store will be created when you first use the chat interface.")    # Test Vector Store Query
    st.header("🔍 Test Vector Store Query")

    if os.path.exists(vector_store_path):
        test_query = st.text_input("Enter a test query:", "penyakit langka")

        if st.button("Test Query") and test_query:
            try:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
                vector_store = Chroma(
                    persist_directory=vector_store_path,
                    embedding_function=embeddings
                )

                # Create retriever
                retriever = vector_store.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={
                        "k": 5,
                        "score_threshold": 0.3
                    }
                )

                # Get relevant documents
                docs = retriever.get_relevant_documents(test_query)

                st.success(f"Found {len(docs)} relevant documents")

                for i, doc in enumerate(docs):
                    with st.expander(f"Result {i+1} - Score: {getattr(doc, 'score', 'N/A')}"):
                        st.write("**Source:**", doc.metadata.get('source', 'Unknown'))
                        st.write("**Type:**", doc.metadata.get('document_type', 'Unknown'))
                        st.write("**Content:**")
                        st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)

            except Exception as e:
                st.error(f"Error testing query: {e}")

    # Performance Metrics
    st.header("⚡ Performance Information")

    if 'load_time' in st.session_state:
        st.metric("Last Load Time", f"{st.session_state.load_time:.2f} seconds")

    # Actions
    st.header("🛠️ Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Force Rebuild Vector Store"):
            st.info("Rebuilding vector store from documents...")

            try:
                # Clear Streamlit cache to force reload
                st.cache_resource.clear()

                # Remove existing vector store
                import shutil
                if os.path.exists(vector_store_path):
                    shutil.rmtree(vector_store_path)
                    st.success("✅ Old vector store deleted")

                # Rebuild vector store immediately
                with st.spinner("Building new vector store..."):
                    # Initialize embeddings
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

                    # Load all documents
                    all_docs = []
                    md_files = glob.glob('./data/**/*.md', recursive=True)

                    if not md_files:
                        st.error("❌ No markdown files found in data directory")
                        return

                    for file_path in md_files:
                        try:
                            loader = UnstructuredMarkdownLoader(file_path)
                            docs = loader.load()

                            # Add document type metadata
                            doc_type = determine_document_type(file_path)
                            for doc in docs:
                                doc.metadata['document_type'] = doc_type
                                doc.metadata['source_category'] = doc_type
                                doc.metadata['original_source'] = file_path

                            all_docs.extend(docs)
                            st.info(f"📄 Loaded {len(docs)} chunks from {os.path.basename(file_path)}")

                        except Exception as e:
                            st.warning(f"⚠️ Failed to load {file_path}: {e}")
                            continue

                    if all_docs:
                        # Split documents
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                        texts = text_splitter.split_documents(all_docs)

                        # Create new vector store
                        vector_store = Chroma.from_documents(
                            texts,
                            embeddings,
                            persist_directory=vector_store_path
                        )

                        st.success(f"✅ Vector store rebuilt successfully with {len(texts)} document chunks!")
                        st.balloons()

                        # Show summary
                        doc_types = {}
                        for doc in all_docs:
                            doc_type = doc.metadata.get('document_type', 'unknown')
                            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

                        summary = ", ".join([f"{count} {dtype}" for dtype, count in doc_types.items()])
                        st.info(f"📊 Document summary: {summary}")

                    else:
                        st.error("❌ No documents were loaded successfully")

            except Exception as e:
                st.error(f"❌ Error rebuilding vector store: {e}")
                import traceback
                st.code(traceback.format_exc())

    with col2:
        if st.button("🧪 Run Vector Store Test"):
            with st.spinner("Running test..."):
                try:
                    # Run inline test instead of subprocess
                    st.info("🔍 Testing vector store loading...")

                    # Test if we can load the vector store
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

                    if os.path.exists(vector_store_path):
                        vector_store = Chroma(
                            persist_directory=vector_store_path,
                            embedding_function=embeddings
                        )
                        count = vector_store._collection.count()
                        st.success(f"✅ Vector store loaded successfully with {count} documents")

                        if count > 0:
                            # Test a simple query
                            retriever = vector_store.as_retriever(search_kwargs={"k": 1})
                            docs = retriever.get_relevant_documents("test")
                            st.success(f"✅ Query test successful - found {len(docs)} results")
                        else:
                            st.warning("⚠️ Vector store is empty")
                    else:
                        st.error("❌ Vector store does not exist")

                except Exception as e:
                    st.error(f"❌ Test failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    with col3:
        if st.button("📊 Show Memory Usage"):
            try:
                import psutil
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                st.metric("Memory Usage", f"{memory_info.rss / 1024 / 1024:.1f} MB")

                # Additional system info
                st.write(f"**CPU Usage:** {psutil.cpu_percent()}%")
                disk_usage = psutil.disk_usage('/')
                st.write(f"**Disk Usage:** {disk_usage.percent}%")

            except ImportError:
                st.warning("psutil not available for memory monitoring")
            except Exception as e:
                st.error(f"Error getting memory info: {e}")

    # Add container-specific information
    st.header("🐳 Container Information")
    if os.path.exists('/.dockerenv'):
        st.info("🐳 Running inside Docker container")

        # Show mounted volumes info
        with st.expander("📁 Volume Information"):
            st.write("**Data Directory:** `/app/data` (mounted read-only)")
            st.write("**Vector Store:** `/app/chroma_db` (persistent volume)")

            # Check if data mount is working
            data_mount_works = os.path.exists('./data') and len(glob.glob('./data/**/*.md', recursive=True)) > 0
            st.write(f"**Data Mount Status:** {'✅ Working' if data_mount_works else '❌ Not working'}")

    else:
        st.info("💻 Running in local environment")

def show_chat_page(qa_chain):
    """Display the main chat interface"""
    # Add logo and title
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("# 🏥 Rare Disease Helper Chatbot")

    st.markdown("---")

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

            # Display sources with document types for transparency
            sources = response['source_documents']
            source_list = "\n\n**Sumber Informasi:**\n"

            # Get unique sources with types to avoid duplicates
            unique_sources = {}
            for doc in sources:
                source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                doc_type = doc.metadata.get('document_type', 'general')

                # Map document types to Indonesian display names
                type_display = {
                    'medical_reference': '📚 Referensi Medis',
                    'user_story': '👥 Pengalaman Keluarga',
                    'community_resource': '🏘️ Sumber Komunitas',
                    'clinical_guideline': '⚕️ Pedoman Klinis',
                    'general': '📄 Umum'
                }

                display_type = type_display.get(doc_type, '📄 Umum')
                unique_sources[source_name] = display_type

            for source_name, source_type in sorted(unique_sources.items()):
                source_list += f"- {source_type}: {source_name}\n"

            full_response = answer + source_list

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- Main Application Logic ---
# Add title and logo here
st.set_page_config(
    page_title="Rare Disease Helper Chatbot",
    page_icon="🏥",
    layout="wide"
)

st.sidebar.title("🏥 Menu")
st.sidebar.markdown("Navigate through the chatbot features")

# Navigation
page = st.sidebar.selectbox(
    "Select Page",
    ["💬 Chat", "🔧 Debug Info"],
    index=0
)

# Add logo and title
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Option 1: If you have a logo file
    # st.image("logo.png", width=200)

    # Option 2: Using emoji as logo
    st.markdown("# 🏥 Rare Disease Helper Chatbot")

    # Option 3: Custom HTML styling
    # st.markdown("""
    # <div style="text-align: center;">
    #     <h1>🏥 Rare Disease Helper Chatbot</h1>
    #     <p style="font-size: 18px; color: #666;">Your AI assistant for rare disease information</p>
    # </div>
    # """, unsafe_allow_html=True)

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
elif page == "💬 Chat":
    show_chat_page(qa_chain)
