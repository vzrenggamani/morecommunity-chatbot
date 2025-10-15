import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .document_utils import determine_document_type, get_data_directory
from .vectorstore_utils import should_rebuild_vectorstore


@st.cache_resource
def load_llm_and_retriever():
    """
    Loads components with persistent vector store to minimize cold starts.
    """

    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Smart caching: Check if vector store needs rebuilding
    vector_store_path = "chroma_db"
    needs_rebuild, reason = should_rebuild_vectorstore(vector_store_path)

    if not needs_rebuild:
        st.info("📚 Memuat vector store yang sudah ada...")
        try:
            # Load existing vector store
            vector_store = Chroma(
                persist_directory=vector_store_path, embedding_function=embeddings
            )
            st.success(f"✅ Vector store berhasil dimuat dari cache! ({reason})")
        except Exception as e:
            st.warning(f"⚠️ Gagal memuat vector store cache: {e}")
            needs_rebuild = True
            reason = "Cache corrupted"

    # Rebuild if needed
    if needs_rebuild:
        vector_store = _rebuild_vector_store(embeddings, vector_store_path, reason)

    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        temperature=0.7,
        max_output_tokens=1024,
    )

    # Create custom prompt template
    custom_prompt_template = """
    Anda adalah dokter AI yang ahli dalam bidang penyakit langka dan kesehatan. Anda memiliki akses ke berbagai jenis pengetahuan medis:

    KONTEKS UMUM (bukan data pasien saat ini):
    {context}

    Ketika memberikan jawaban, sesuaikan pendekatan berdasarkan jenis informasi:
    - Informasi medis/klinis: Gunakan dengan otoritas profesional dan presisi klinis
    - Pengalaman pasien: Gunakan untuk memberikan empati dan pemahaman mendalam tentang pengalaman keluarga melalui cerita yang dialami oleh pasien sebelumnya.
    - Sumber komunitas: Gunakan untuk memberikan dukungan dan informasi praktis tentang resources

    PERTANYAAN PASIEN: {question}

    PENTING - JANGAN LAKUKAN:
    - Informasi dalam konteks hanya sebagai referensi umum, BUKAN bagian dari riwayat pasien.
    - JANGAN mengasumsikan kondisi atau gejala pasien kecuali dinyatakan dalam pertanyaan.
    - JANGAN katakan "berdasarkan cerita Anda", "dari informasi yang Anda berikan"
    - JANGAN rujuk ke sumber atau dokumen secara eksplisit

    LAKUKAN:
    - Pisahkan antara penjelasan medis umum dan rekomendasi langkah praktis bagi pasien.
    - Jika pertanyaan tidak menyertakan gejala, berikan panduan umum untuk proses evaluasi medis.
    - Jawab langsung dan profesional seperti dokter berpengalaman
    - Integrasikan semua jenis pengetahuan secara natural
    - Berikan empati berdasarkan pengalaman keluarga yang Anda ketahui
    - Sertakan informasi praktis dan dukungan komunitas jika relevan
    - Selalu sarankan konsultasi medis profesional untuk diagnosis dan pengobatan

    JAWABAN DOKTER:"""

    PROMPT = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )

    # Create the RetrievalQA chain with custom prompt
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,  # Reduced from 5 to 3 to save tokens
            "score_threshold": 0.3,  # Lower threshold = more lenient matching
        },
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    return qa_chain


def _rebuild_vector_store(embeddings, vector_store_path, reason):
    """Rebuild vector store from documents"""
    st.info(f"🔄 Membangun ulang vector store: {reason}")

    # 1. Load Documents with type labeling
    st.info("📖 Memuat dokumen dari direktori 'data'...")

    # Get the correct data directory
    data_dir, md_files = get_data_directory()

    if not data_dir or not md_files:
        st.error("❌ Tidak ada file markdown ditemukan!")
        _show_debug_info()
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
                doc.metadata["document_type"] = doc_type
                doc.metadata["source_category"] = doc_type
                # Keep original source path
                doc.metadata["original_source"] = file_path

            all_docs.extend(docs)
            st.info(
                f"✅ Memuat {len(docs)} dokumen dari {doc_type}: {os.path.basename(file_path)}"
            )

        except Exception as e:
            # If a file fails, print a warning and skip it
            st.warning(f"❌ Tidak dapat memuat file: {file_path}. Error: {e}")
            continue  # Go to the next file

    if not all_docs:
        st.error(
            "Tidak ada dokumen yang dimuat. Silakan periksa folder 'data' dan isi file."
        )
        st.stop()

    # Show summary of loaded documents by type
    doc_types = {}
    for doc in all_docs:
        doc_type = doc.metadata.get("document_type", "unknown")
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

    summary = ", ".join([f"{count} {dtype}" for dtype, count in doc_types.items()])
    st.success(f"✅ Berhasil memuat {len(all_docs)} dokumen total: {summary}")

    # 2. Split Documents into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(all_docs)

    # 3. Create Embeddings and Store in ChromaDB
    st.info("🔄 Membuat embeddings dan menyimpan ke vector store...")
    vector_store = Chroma.from_documents(
        texts, embeddings, persist_directory=vector_store_path
    )
    st.success("✅ Vector store baru berhasil dibuat dan disimpan!")

    return vector_store


def _show_debug_info():
    """Show debugging information when files are not found"""
    st.info("🔍 **Debugging info:**")
    st.info(f"📍 Working directory: {os.getcwd()}")
    st.info(f"📍 Vector store path: chroma_db")
    st.info(f"📍 Data directory exists: {os.path.exists('./data')}")

    # List what's actually in the current directory
    try:
        current_contents = os.listdir(".")
        st.info(f"📁 Current directory contents: {current_contents}")

        if os.path.exists("./data"):
            data_contents = os.listdir("./data")
            st.info(f"📁 Data directory contents: {data_contents}")

            # Check subdirectories
            for item in data_contents:
                item_path = os.path.join("./data", item)
                if os.path.isdir(item_path):
                    subdir_contents = os.listdir(item_path)
                    st.info(f"📁 {item}/ contents: {subdir_contents}")
        else:
            st.info("📁 Data directory does not exist")

    except Exception as e:
        st.error(f"Error listing directory contents: {e}")
