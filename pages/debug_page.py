import streamlit as st
import os
import sys
import glob
import pandas as pd
import shutil
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.document_utils import determine_document_type, get_data_directory
from utils.vectorstore_utils import should_rebuild_vectorstore


def show_debug_page():
    """Display debug information about the vector store and system"""
    st.title("🔧 Debug Information")
    st.markdown(
        "This page shows technical information about the system for debugging purposes."
    )

    # System Information
    _show_system_information()

    # Document Information
    _show_document_information()

    # Vector Store Information
    _show_vector_store_information()

    # Test Vector Store Query
    _show_test_query_section()

    # Performance Metrics
    _show_performance_metrics()

    # Token Usage Analytics
    _show_token_analytics()

    # Container Information
    _show_container_information()

    # Actions
    _show_actions()


def _show_system_information():
    """Show system information section"""
    st.header("📊 System Information")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Environment")
        st.write(f"**Python Version:** {sys.version.split()[0]}")
        st.write(f"**Current Working Directory:** {os.getcwd()}")
        st.write(
            f"**Google API Key Set:** {'✅ Yes' if os.getenv('GOOGLE_API_KEY') else '❌ No'}"
        )

    with col2:
        st.subheader("Directories")
        data_dir = "./data"
        vector_store_dir = "chroma_db"
        st.write(
            f"**Data Directory Exists:** {'✅ Yes' if os.path.exists(data_dir) else '❌ No'}"
        )
        st.write(
            f"**Vector Store Exists:** {'✅ Yes' if os.path.exists(vector_store_dir) else '❌ No'}"
        )


def _show_document_information():
    """Show document information section"""
    st.header("📄 Document Information")

    # Check markdown files
    md_files = glob.glob("./data/**/*.md", recursive=True)
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
            with st.expander(
                f"📁 {doc_type.replace('_', ' ').title()} ({len(files)} files)"
            ):
                for file_path in files:
                    file_name = os.path.basename(file_path)
                    file_size = (
                        os.path.getsize(file_path) if os.path.exists(file_path) else 0
                    )
                    file_modified = (
                        datetime.fromtimestamp(os.path.getmtime(file_path)).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                        if os.path.exists(file_path)
                        else "Unknown"
                    )

                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**{file_name}**")
                    with col2:
                        st.write(f"{file_size} bytes")
                    with col3:
                        st.write(file_modified)
    else:
        st.warning("No markdown files found in data directory")
        _show_directory_debug_info()


def _show_directory_debug_info():
    """Show directory debugging information"""
    st.info("🔍 **Debugging info:**")
    st.info(f"📍 Working directory: {os.getcwd()}")

    # Check various possible data directory locations
    possible_paths = ["./data", "/app/data", "data"]
    for path in possible_paths:
        exists = os.path.exists(path)
        st.info(f"📁 {path} exists: {'✅' if exists else '❌'}")
        if exists:
            try:
                contents = os.listdir(path)
                st.info(f"📁 {path} contents: {contents}")
            except Exception as e:
                st.info(f"📁 {path} error reading contents: {e}")


def _show_vector_store_information():
    """Show vector store information section"""
    st.header("🗄️ Vector Store Information")

    vector_store_path = "chroma_db"

    # Check both if directory exists and if it contains valid ChromaDB files
    vector_store_exists = os.path.exists(vector_store_path)
    has_chroma_files = False

    if vector_store_exists:
        # Check for ChromaDB specific files
        chroma_files = ["chroma.sqlite3", "chroma.sqlite3-wal", "chroma.sqlite3-shm"]
        has_chroma_files = any(
            os.path.exists(os.path.join(vector_store_path, f)) for f in chroma_files
        )

    if vector_store_exists:
        if has_chroma_files:
            st.success("Vector store directory exists with ChromaDB files")
        else:
            st.warning("Vector store directory exists but appears empty or invalid")

        try:
            # Load vector store to get information
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            vector_store = Chroma(
                persist_directory=vector_store_path, embedding_function=embeddings
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
                    vector_store_modified = datetime.fromtimestamp(
                        os.path.getmtime(vector_store_path)
                    ).strftime("%Y-%m-%d %H:%M:%S")
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
                _show_chromadb_details(collection, vector_store_path)

            # Sample documents
            if st.button("Show Sample Documents"):
                _show_sample_documents(collection, count)

        except Exception as e:
            st.error(f"Error loading vector store: {e}")
            st.info(
                "This usually means the vector store is corrupted or was not properly created."
            )
    else:
        st.error("Vector store directory does not exist")
        st.info(
            "The vector store will be created when you first use the chat interface."
        )


def _show_chromadb_details(collection, vector_store_path):
    """Show ChromaDB technical details"""
    try:
        st.write(f"**Collection Name:** {collection.name}")
        st.write(f"**Collection ID:** {collection.id}")

        # List files in vector store directory
        st.write("**Files in vector store directory:**")
        for root, dirs, files in os.walk(vector_store_path):
            level = root.replace(vector_store_path, "").count(os.sep)
            indent = " " * 2 * level
            st.code(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files:
                file_path = os.path.join(root, file)
                size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                st.code(f"{subindent}{file} ({size} bytes)")

    except Exception as e:
        st.error(f"Error getting ChromaDB details: {e}")


def _show_sample_documents(collection, count):
    """Show sample documents from the vector store"""
    if count > 0:
        try:
            # Get a few sample documents
            results = collection.peek(limit=3)

            st.subheader("Sample Document Chunks")
            for i, (doc_id, metadata, document) in enumerate(
                zip(results["ids"], results["metadatas"], results["documents"])
            ):
                with st.expander(f"Document {i+1} - {doc_id[:50]}..."):
                    st.write("**Metadata:**")
                    st.json(metadata)
                    st.write("**Content Preview:**")
                    st.write(
                        document[:500] + "..." if len(document) > 500 else document
                    )

        except Exception as e:
            st.error(f"Error retrieving sample documents: {e}")
    else:
        st.warning("No documents to show - vector store is empty")


def _show_test_query_section():
    """Show test query section"""
    st.header("🔍 Test Vector Store Query")

    vector_store_path = "chroma_db"
    if os.path.exists(vector_store_path):
        test_query = st.text_input("Enter a test query:", "penyakit langka")

        if st.button("Test Query") and test_query:
            try:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004"
                )
                vector_store = Chroma(
                    persist_directory=vector_store_path, embedding_function=embeddings
                )

                # Create retriever
                retriever = vector_store.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={"k": 5, "score_threshold": 0.3},
                )

                # Get relevant documents
                docs = retriever.get_relevant_documents(test_query)

                st.success(f"Found {len(docs)} relevant documents")

                for i, doc in enumerate(docs):
                    with st.expander(
                        f"Result {i+1} - Score: {getattr(doc, 'score', 'N/A')}"
                    ):
                        st.write("**Source:**", doc.metadata.get("source", "Unknown"))
                        st.write(
                            "**Type:**", doc.metadata.get("document_type", "Unknown")
                        )
                        st.write("**Content:**")
                        st.write(
                            doc.page_content[:500] + "..."
                            if len(doc.page_content) > 500
                            else doc.page_content
                        )

            except Exception as e:
                st.error(f"Error testing query: {e}")


def _show_performance_metrics():
    """Show performance metrics"""
    st.header("⚡ Performance Information")

    if "load_time" in st.session_state:
        st.metric("Last Load Time", f"{st.session_state.load_time:.2f} seconds")


def _show_token_analytics():
    """Show token usage analytics"""
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
            total_tokens = usage["total_input_tokens"] + usage["total_output_tokens"]
            st.metric("Total Tokens", f"{total_tokens:,}")

        # Cost estimation
        estimated_cost = (usage["total_input_tokens"] * 0.000001) + (
            usage["total_output_tokens"] * 0.000002
        )
        st.metric("Estimated Session Cost", f"${estimated_cost:.6f}")

        # Conversation history
        if usage["session_history"]:
            _show_conversation_history(usage)

        # Additional token statistics
        if usage["conversation_count"] > 0:
            _show_average_token_usage(usage)

        # Reset button for token tracking
        if st.button("🔄 Reset Token Usage Statistics"):
            st.session_state.token_usage = {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "conversation_count": 0,
                "session_history": [],
            }
            st.success("✅ Token usage statistics reset!")
            st.rerun()
    else:
        _show_token_usage_info()


def _show_conversation_history(usage):
    """Show conversation history table"""
    st.subheader("📝 Recent Conversation Token Usage")

    # Create a table of recent conversations
    history_data = []
    for entry in reversed(usage["session_history"]):  # Show most recent first
        history_data.append(
            {
                "Time": entry["timestamp"],
                "Query Preview": entry["query_preview"],
                "Input Tokens": f"{entry['input_tokens']:,}",
                "Output Tokens": f"{entry['output_tokens']:,}",
                "Total": f"{entry['input_tokens'] + entry['output_tokens']:,}",
            }
        )

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

        chart_data = pd.DataFrame(
            {
                "Conversation": range(1, len(timestamps) + 1),
                "Input Tokens": input_tokens,
                "Output Tokens": output_tokens,
            }
        )

        st.line_chart(chart_data.set_index("Conversation"))


def _show_average_token_usage(usage):
    """Show average token usage statistics"""
    avg_input = usage["total_input_tokens"] / usage["conversation_count"]
    avg_output = usage["total_output_tokens"] / usage["conversation_count"]

    st.subheader("📊 Average Token Usage per Conversation")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Avg Input Tokens", f"{avg_input:.0f}")
    with col2:
        st.metric("Avg Output Tokens", f"{avg_output:.0f}")
    with col3:
        st.metric("Avg Total Tokens", f"{avg_input + avg_output:.0f}")


def _show_token_usage_info():
    """Show information about token usage tracking"""
    st.info(
        "💡 No token usage data available yet. Start a conversation in the Chat tab to see token analytics."
    )
    st.write("**What you'll see here:**")
    st.write("- Total tokens used (input/output)")
    st.write("- Estimated costs")
    st.write("- Per-conversation token breakdown")
    st.write("- Token usage trends over time")


def _show_container_information():
    """Show container-specific information"""
    st.header("🐳 Container Information")
    if os.path.exists("/.dockerenv"):
        st.info("🐳 Running inside Docker container")

        # Show mounted volumes info
        with st.expander("📁 Volume Information"):
            st.write("**Data Directory:** `/app/data` (mounted read-only)")
            st.write("**Vector Store:** `/app/chroma_db` (persistent volume)")

            # Check if data mount is working
            data_mount_works = (
                os.path.exists("./data")
                and len(glob.glob("./data/**/*.md", recursive=True)) > 0
            )
            st.write(
                f"**Data Mount Status:** {'✅ Working' if data_mount_works else '❌ Not working'}"
            )

    else:
        st.info("💻 Running in local environment")


def _show_actions():
    """Show action buttons section"""
    st.header("🛠️ Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Force Rebuild Vector Store"):
            _rebuild_vector_store()

    with col2:
        if st.button("🧪 Run Vector Store Test"):
            _run_vector_store_test()

    with col3:
        if st.button("📊 Show Memory Usage"):
            _show_memory_usage()


def _rebuild_vector_store():
    """Rebuild vector store from documents"""
    vector_store_path = "chroma_db"
    st.info("Rebuilding vector store from documents...")

    try:
        # Clear Streamlit cache to force reload
        st.cache_resource.clear()

        # Remove existing vector store
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
                st.info(
                    "🔍 Please check that your data files are properly mounted in the Docker container"
                )
                return

            st.info(f"📁 Using data directory: {data_dir}")

            for file_path in md_files:
                try:
                    loader = UnstructuredMarkdownLoader(file_path)
                    docs = loader.load()

                    # Add document type metadata
                    doc_type = determine_document_type(file_path)
                    for doc in docs:
                        doc.metadata["document_type"] = doc_type
                        doc.metadata["source_category"] = doc_type
                        doc.metadata["original_source"] = file_path

                    all_docs.extend(docs)
                    st.info(
                        f"📄 Loaded {len(docs)} chunks from {os.path.basename(file_path)}"
                    )

                except Exception as e:
                    st.warning(f"⚠️ Failed to load {file_path}: {e}")
                    continue

            if all_docs:
                # Split documents
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=100
                )
                texts = text_splitter.split_documents(all_docs)

                # Create new vector store
                vector_store = Chroma.from_documents(
                    texts, embeddings, persist_directory=vector_store_path
                )

                st.success(
                    f"✅ Vector store rebuilt successfully with {len(texts)} document chunks!"
                )
                st.balloons()

                # Show summary
                doc_types = {}
                for doc in all_docs:
                    doc_type = doc.metadata.get("document_type", "unknown")
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

                summary = ", ".join(
                    [f"{count} {dtype}" for dtype, count in doc_types.items()]
                )
                st.info(f"📊 Document summary: {summary}")

            else:
                st.error("❌ No documents were loaded successfully")

    except Exception as e:
        st.error(f"❌ Error rebuilding vector store: {e}")
        import traceback

        st.code(traceback.format_exc())


def _run_vector_store_test():
    """Run vector store test"""
    vector_store_path = "chroma_db"
    with st.spinner("Running test..."):
        try:
            # Run inline test instead of subprocess
            st.info("🔍 Testing vector store loading...")

            # Test if we can load the vector store
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

            if os.path.exists(vector_store_path):
                vector_store = Chroma(
                    persist_directory=vector_store_path, embedding_function=embeddings
                )
                count = vector_store._collection.count()
                st.success(
                    f"✅ Vector store loaded successfully with {count} documents"
                )

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


def _show_memory_usage():
    """Show memory usage information"""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        st.metric("Memory Usage", f"{memory_info.rss / 1024 / 1024:.1f} MB")

        # Additional system info
        st.write(f"**CPU Usage:** {psutil.cpu_percent()}%")
        disk_usage = psutil.disk_usage("/")
        st.write(f"**Disk Usage:** {disk_usage.percent}%")

    except ImportError:
        st.warning("psutil not available for memory monitoring")
    except Exception as e:
        st.error(f"Error getting memory info: {e}")
