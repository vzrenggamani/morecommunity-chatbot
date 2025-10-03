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
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import glob
import tiktoken
import json
import pandas as pd

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

def count_tokens(text, model_name="gpt-3.5-turbo"):
    """Count tokens in text using tiktoken"""
    try:
        # Use a general tokenizer for Gemini models
        encoding = tiktoken.get_encoding("cl100k_base")  # This is used by GPT-4 and similar models
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        # Fallback: rough estimation (1 token ≈ 4 characters for most languages)
        return len(text) // 4

def initialize_token_tracking():
    """Initialize token usage tracking in session state"""
    if "token_usage" not in st.session_state:
        st.session_state.token_usage = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "conversation_count": 0,
            "session_history": []
        }

def add_token_usage(input_tokens, output_tokens, query_text, response_text):
    """Add token usage to session tracking"""
    if "token_usage" not in st.session_state:
        initialize_token_tracking()

    st.session_state.token_usage["total_input_tokens"] += input_tokens
    st.session_state.token_usage["total_output_tokens"] += output_tokens
    st.session_state.token_usage["conversation_count"] += 1

    # Add to history (keep last 10 conversations)
    conversation_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "query_preview": query_text[:50] + "..." if len(query_text) > 50 else query_text,
        "response_preview": response_text[:50] + "..." if len(response_text) > 50 else response_text
    }

    st.session_state.token_usage["session_history"].append(conversation_entry)

    # Keep only last 10 conversations
    if len(st.session_state.token_usage["session_history"]) > 10:
        st.session_state.token_usage["session_history"] = st.session_state.token_usage["session_history"][-10:]

def format_token_info(input_tokens, output_tokens, context_text, num_retrieved_docs):
    """Format token usage information for display"""
    total_tokens = input_tokens + output_tokens

    # Estimate cost (using approximate pricing for Gemini Pro)
    input_cost = (input_tokens / 1000) * 0.00025  # $0.00025 per 1K input tokens
    output_cost = (output_tokens / 1000) * 0.0005  # $0.0005 per 1K output tokens
    total_cost = input_cost + output_cost

    context_tokens = count_tokens(context_text) if context_text else 0

    token_info = f"""

**📊 Token Usage:**
- Input: {input_tokens:,} tokens
- Output: {output_tokens:,} tokens
- Total: {total_tokens:,} tokens
- Context: {context_tokens:,} tokens (from {num_retrieved_docs} docs)
- Estimated cost: ${total_cost:.6f}

**📈 Session Stats:**
- Total conversations: {st.session_state.token_usage['conversation_count']}
- Total input tokens: {st.session_state.token_usage['total_input_tokens']:,}
- Total output tokens: {st.session_state.token_usage['total_output_tokens']:,}
- Session cost: ${(st.session_state.token_usage['total_input_tokens'] * 0.00025 + st.session_state.token_usage['total_output_tokens'] * 0.0005) / 1000:.6f}
"""
    return token_info

def get_optimized_context(relevant_docs, max_tokens=500):
    """Get context text optimized for token usage"""
    context_parts = []
    current_tokens = 0

    for i, doc in enumerate(relevant_docs):
        doc_content = doc.page_content
        doc_tokens = count_tokens(doc_content)

        # If adding this document would exceed limit, truncate it
        if current_tokens + doc_tokens > max_tokens:
            remaining_tokens = max_tokens - current_tokens
            if remaining_tokens > 50:  # Only add if we have meaningful space left
                # Truncate the document to fit
                words = doc_content.split()
                truncated_content = ""
                for word in words:
                    test_content = truncated_content + " " + word if truncated_content else word
                    if count_tokens(test_content) <= remaining_tokens:
                        truncated_content = test_content
                    else:
                        break
                if truncated_content:
                    context_parts.append(truncated_content + "...")
            break
        else:
            context_parts.append(doc_content)
            current_tokens += doc_tokens

    return "\n\n".join(context_parts), len(context_parts)
    """Format token usage information for display"""
    total_tokens = input_tokens + output_tokens

    # Rough cost estimation (this varies by provider, adjust as needed)
    # These are approximate costs for reference
    estimated_cost_input = input_tokens * 0.000001  # $0.001 per 1K tokens (example)
    estimated_cost_output = output_tokens * 0.000002  # $0.002 per 1K tokens (example)
    total_estimated_cost = estimated_cost_input + estimated_cost_output

    context_info = ""
    if context_size and num_docs:
        context_tokens = count_tokens(context_size) if isinstance(context_size, str) else context_size
        context_info = f"\n- Context: {context_tokens:,} tokens ({num_docs} docs)"

    return f"""
📊 **Token Usage:**
- Input tokens: {input_tokens:,}{context_info}
- Output tokens: {output_tokens:,}
- Total tokens: {total_tokens:,}
- Estimated cost: ${total_estimated_cost:.6f} (approximate)
"""@st.cache_resource
def get_data_directory():
    """Get the correct data directory path for different environments"""
    # Try different possible data directory locations
    possible_paths = [
        './data',           # Local development
        '/app/data',        # Docker container
        'data',             # Alternative local
        os.path.join(os.getcwd(), 'data')  # Absolute path from current working directory
    ]

    for path in possible_paths:
        if os.path.exists(path):
            # Check if it actually contains markdown files
            md_files = glob.glob(f'{path}/**/*.md', recursive=True)
            if md_files:
                return path, md_files

    return None, []

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

        # Get the correct data directory
        data_dir, md_files = get_data_directory()

        if not data_dir or not md_files:
            st.error("❌ Tidak ada file markdown ditemukan!")
            st.info("🔍 **Debugging info:**")
            st.info(f"📍 Working directory: {os.getcwd()}")
            st.info(f"📍 Vector store path: {vector_store_path}")
            st.info(f"📍 Data directory exists: {os.path.exists('./data')}")

            # List what's actually in the current directory
            try:
                current_contents = os.listdir('.')
                st.info(f"📁 Current directory contents: {current_contents}")

                if os.path.exists('./data'):
                    data_contents = os.listdir('./data')
                    st.info(f"📁 Data directory contents: {data_contents}")

                    # Check subdirectories
                    for item in data_contents:
                        item_path = os.path.join('./data', item)
                        if os.path.isdir(item_path):
                            subdir_contents = os.listdir(item_path)
                            st.info(f"📁 {item}/ contents: {subdir_contents}")
                else:
                    st.info("📁 Data directory does not exist")

            except Exception as e:
                st.error(f"Error listing directory contents: {e}")

            st.stop()

        st.success(f"✅ Found {len(md_files)} markdown files in {data_dir}")
        all_docs = []

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
            "k": 3,  # Reduced from 5 to 3 to save tokens
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
        st.info("🔍 **Debugging info:**")
        st.info(f"📍 Working directory: {os.getcwd()}")

        # Check various possible data directory locations
        possible_paths = ['./data', '/app/data', 'data']
        for path in possible_paths:
            exists = os.path.exists(path)
            st.info(f"📁 {path} exists: {'✅' if exists else '❌'}")
            if exists:
                try:
                    contents = os.listdir(path)
                    st.info(f"📁 {path} contents: {contents}")
                except Exception as e:
                    st.info(f"📁 {path} error reading contents: {e}")

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

    # Token Usage Analytics
    st.header("📊 Token Usage Analytics")

    if st.session_state.get("token_usage"):
        usage = st.session_state.token_usage

        # Overall statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Conversations", usage["conversation_count"])
        with col2:
            st.metric("Input Tokens", f"{usage['total_input_tokens']:,}")
        with col3:
            st.metric("Output Tokens", f"{usage['total_output_tokens']:,}")
        with col4:
            total_tokens = usage['total_input_tokens'] + usage['total_output_tokens']
            st.metric("Total Tokens", f"{total_tokens:,}")

        # Cost estimation
        estimated_cost = (usage['total_input_tokens'] * 0.000001) + (usage['total_output_tokens'] * 0.000002)
        st.metric("Estimated Session Cost", f"${estimated_cost:.6f}")

        # Conversation history
        if usage["session_history"]:
            st.subheader("📝 Recent Conversation Token Usage")

            # Create a table of recent conversations
            history_data = []
            for entry in reversed(usage["session_history"]):  # Show most recent first
                history_data.append({
                    "Time": entry["timestamp"],
                    "Query Preview": entry["query_preview"],
                    "Input Tokens": f"{entry['input_tokens']:,}",
                    "Output Tokens": f"{entry['output_tokens']:,}",
                    "Total": f"{entry['input_tokens'] + entry['output_tokens']:,}"
                })

            # Display as table
            df = pd.DataFrame(history_data)
            st.dataframe(df, use_container_width=True)

            # Show token usage trend if we have multiple conversations
            if len(usage["session_history"]) > 1:
                st.subheader("📈 Token Usage Trend")

                # Prepare data for chart
                timestamps = [entry["timestamp"] for entry in usage["session_history"]]
                input_tokens = [entry["input_tokens"] for entry in usage["session_history"]]
                output_tokens = [entry["output_tokens"] for entry in usage["session_history"]]

                chart_data = pd.DataFrame({
                    "Conversation": range(1, len(timestamps) + 1),
                    "Input Tokens": input_tokens,
                    "Output Tokens": output_tokens
                })

                st.line_chart(chart_data.set_index("Conversation"))

        # Additional token statistics
        if usage["conversation_count"] > 0:
            avg_input = usage['total_input_tokens'] / usage["conversation_count"]
            avg_output = usage['total_output_tokens'] / usage["conversation_count"]

            st.subheader("📊 Average Token Usage per Conversation")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Avg Input Tokens", f"{avg_input:.0f}")
            with col2:
                st.metric("Avg Output Tokens", f"{avg_output:.0f}")
            with col3:
                st.metric("Avg Total Tokens", f"{avg_input + avg_output:.0f}")

        # Reset button for token tracking
        if st.button("🔄 Reset Token Usage Statistics"):
            st.session_state.token_usage = {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "conversation_count": 0,
                "session_history": []
            }
            st.success("✅ Token usage statistics reset!")
            st.rerun()
    else:
        st.info("💡 No token usage data available yet. Start a conversation in the Chat tab to see token analytics.")
        st.write("**What you'll see here:**")
        st.write("- Total tokens used (input/output)")
        st.write("- Estimated costs")
        st.write("- Per-conversation token breakdown")
        st.write("- Token usage trends over time")

    # Token Usage Configuration
    st.header("⚙️ Token Usage Configuration")

    st.info("💡 **Tip**: Reducing max context tokens will lower input token usage but may reduce answer quality.")

    # Add configuration sliders
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Current Settings")
        st.write("- **Max Context Tokens**: 400")
        st.write("- **Retrieved Documents**: 3")
        st.write("- **Document Chunk Size**: 1000 chars")

    with col2:
        st.subheader("Token Breakdown")
        st.write("- **System Prompt**: ~150 tokens")
        st.write("- **Context**: ~400 tokens (optimized)")
        st.write("- **User Question**: varies")
        st.write("- **Total Input**: ~550-600 tokens")

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

                    # Load all documents using robust data directory detection
                    all_docs = []
                    data_dir, md_files = get_data_directory()

                    if not md_files:
                        st.error("❌ No markdown files found in data directory")
                        st.info("🔍 Please check that your data files are properly mounted in the Docker container")
                        return

                    st.info(f"📁 Using data directory: {data_dir}")

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
    # Initialize token tracking
    initialize_token_tracking()

    # Add logo and title
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("# 🏥 Rare Disease Helper Chatbot")

    st.markdown("---")

    # Display token usage summary in sidebar
    with st.sidebar:
        if st.session_state.get("token_usage"):
            st.markdown("### 📊 Token Usage (This Session)")
            usage = st.session_state.token_usage
            st.metric("Total Conversations", usage["conversation_count"])
            st.metric("Total Input Tokens", f"{usage['total_input_tokens']:,}")
            st.metric("Total Output Tokens", f"{usage['total_output_tokens']:,}")
            st.metric("Total Tokens", f"{usage['total_input_tokens'] + usage['total_output_tokens']:,}")

            # Rough cost estimation
            total_cost = (usage['total_input_tokens'] * 0.000001) + (usage['total_output_tokens'] * 0.000002)
            st.metric("Estimated Cost", f"${total_cost:.6f}")

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
            # Calculate input tokens (context + prompt)
            # First, get the context that will be sent to the LLM
            retriever = qa_chain.retriever
            relevant_docs = retriever.get_relevant_documents(prompt)
            context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

            # Debug: Show context statistics
            context_length = len(context_text)
            num_retrieved_docs = len(relevant_docs)

            # Calculate tokens for the full prompt that will be sent to LLM
            full_prompt_text = f"""
            Anda adalah dokter AI yang ahli dalam bidang penyakit langka dan kesehatan. Anda memiliki akses ke berbagai jenis pengetahuan medis:

            SUMBER PENGETAHUAN ANDA:
            {context_text}

            Ketika memberikan jawaban, sesuaikan pendekatan berdasarkan jenis informasi:
            - Informasi medis/klinis: Gunakan dengan otoritas profesional dan presisi klinis
            - Pengalaman pasien: Gunakan untuk memberikan empati dan pemahaman mendalam tentang pengalaman keluarga
            - Sumber komunitas: Gunakan untuk memberikan dukungan dan informasi praktis tentang resources

            PERTANYAAN PASIEN: {prompt}

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

            input_tokens = count_tokens(full_prompt_text)

            # Get the response from the RAG chain
            response = qa_chain.invoke(prompt)
            answer = response['result']

            # Calculate output tokens
            output_tokens = count_tokens(answer)

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

            # Add token usage information
            token_info = format_token_info(input_tokens, output_tokens, context_text, num_retrieved_docs)

            # Create full response with token info
            full_response = answer + source_list + token_info

            # Track token usage
            add_token_usage(input_tokens, output_tokens, prompt, answer)

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
