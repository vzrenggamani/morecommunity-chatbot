#!/usr/bin/env python3
"""
Pre-build Vector Store Script for Rare Disease Helper Chatbot

This script pre-builds the vector store to eliminate cold starts in production.
Run this script whenever you add or modify documents in the data/ folder.

Usage:
    python build_vector_store.py
    python build_vector_store.py --force  # Force rebuild even if up-to-date
"""

# Patch for ChromaDB compatibility (same as app.py)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("✅ SQLite patch applied for ChromaDB compatibility")
except ImportError:
    print("ℹ️  pysqlite3 not available, using system sqlite3")

import os
import sys
import glob
import argparse
from datetime import datetime
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

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

def should_rebuild_vectorstore(vector_store_path, data_path='./data'):
    """Check if vector store needs rebuilding based on file timestamps"""
    if not os.path.exists(vector_store_path):
        return True, "Vector store tidak ditemukan"

    # Get vector store timestamp
    try:
        vectorstore_timestamp = os.path.getmtime(vector_store_path)
    except:
        return True, "Tidak dapat membaca timestamp vector store"

    # Use the get_data_directory function to find markdown files
    data_dir, md_files = get_data_directory()
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

def load_and_process_documents(data_path='./data'):
    """Load and process all markdown documents"""
    print("📖 Memuat dokumen dari direktori 'data'...")

    # Use robust data directory detection
    data_dir, md_files = get_data_directory()

    if not data_dir or not md_files:
        print("❌ Tidak ada file markdown ditemukan!")
        print(f"📍 Working directory: {os.getcwd()}")
        print(f"📍 Vector store path: /app/chroma_db")
        print(f"📍 Data directory exists: {os.path.exists('./data')}")

        # List what's actually in the current directory
        try:
            current_contents = os.listdir('.')
            print(f"📁 Current directory contents: {current_contents}")

            if os.path.exists('./data'):
                data_contents = os.listdir('./data')
                print(f"📁 Data directory contents: {data_contents}")

                # Check subdirectories
                for item in data_contents:
                    item_path = os.path.join('./data', item)
                    if os.path.isdir(item_path):
                        subdir_contents = os.listdir(item_path)
                        print(f"📁 {item}/ contents: {subdir_contents}")
            else:
                print("📁 Data directory does not exist")

        except Exception as e:
            print(f"Error listing directory contents: {e}")

        return None

    print(f"✅ Found {len(md_files)} markdown files in {data_dir}")
    all_docs = []
    doc_counts = {}

    for file_path in md_files:
        try:
            print(f"   Loading: {file_path}")
            loader = UnstructuredMarkdownLoader(file_path)
            docs = loader.load()

            # Add document type metadata
            doc_type = determine_document_type(file_path)
            for doc in docs:
                doc.metadata['document_type'] = doc_type
                doc.metadata['source_category'] = doc_type
                doc.metadata['original_source'] = file_path
                doc.metadata['build_timestamp'] = datetime.now().isoformat()

            all_docs.extend(docs)
            doc_counts[doc_type] = doc_counts.get(doc_type, 0) + len(docs)
            print(f"✅ Memuat {len(docs)} dokumen dari {doc_type}: {os.path.basename(file_path)}")

        except Exception as e:
            print(f"❌ Gagal memuat file {file_path}: {e}")
            import traceback
            print(f"   Error details: {traceback.format_exc()}")
            continue

    if not all_docs:
        print("❌ Tidak ada dokumen yang berhasil dimuat!")
        return None

    # Show summary
    summary = ", ".join([f"{count} {dtype}" for dtype, count in doc_counts.items()])
    print(f"✅ Total dokumen dimuat: {len(all_docs)} ({summary})")

    return all_docs

def build_vector_store(force_rebuild=False):
    """Build or rebuild the vector store"""
    vector_store_path = "chroma_db"

    print(f"📍 Working directory: {os.getcwd()}")
    print(f"📍 Vector store path: {os.path.abspath(vector_store_path)}")
    print(f"📍 Data directory exists: {os.path.exists('./data')}")

    # Check if rebuild is needed
    if not force_rebuild:
        needs_rebuild, reason = should_rebuild_vectorstore(vector_store_path)
        if not needs_rebuild:
            print(f"✅ Vector store sudah up-to-date: {reason}")
            return True
        print(f"🔄 Membangun ulang vector store: {reason}")
    else:
        print("🔄 Force rebuild vector store...")

    # Load documents
    all_docs = load_and_process_documents()
    if not all_docs:
        print("❌ Tidak ada dokumen untuk diproses")
        return False

    try:
        print("📝 Membuat text chunks...")
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        texts = text_splitter.split_documents(all_docs)
        print(f"✅ Dibuat {len(texts)} text chunks")

        print("🧠 Membuat embeddings dengan Google AI...")
        # Initialize embeddings
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            # Test the embeddings with a simple text
            test_embedding = embeddings.embed_query("test")
            print(f"✅ Embeddings initialized successfully (dimension: {len(test_embedding)})")
        except Exception as e:
            print(f"❌ Failed to initialize embeddings: {e}")
            raise

        # Remove existing vector store if it exists
        if os.path.exists(vector_store_path):
            try:
                import shutil
                # Try to remove it, but handle the case where it's mounted and locked
                shutil.rmtree(vector_store_path)
                print("🗑️ Vector store lama telah dihapus")
            except OSError as e:
                if e.errno == 16:  # Device or resource busy
                    print("⚠️ Vector store directory locked (Docker volume), clearing contents instead...")
                    # Instead of removing the directory, clear its contents
                    for root, dirs, files in os.walk(vector_store_path):
                        for file in files:
                            try:
                                os.remove(os.path.join(root, file))
                            except:
                                pass
                        for dir_name in dirs:
                            try:
                                shutil.rmtree(os.path.join(root, dir_name))
                            except:
                                pass
                    print("🗑️ Vector store contents cleared")
                else:
                    raise

        print("💾 Menyimpan ke ChromaDB...")
        # Create and persist vector store
        vector_store = Chroma.from_documents(
            texts,
            embeddings,
            persist_directory=vector_store_path
        )

        # Verify the vector store was created successfully
        collection = vector_store._collection
        document_count = collection.count()
        print(f"✅ Verifikasi: {document_count} dokumen tersimpan di vector store")

        print(f"✅ Vector store berhasil dibuat di: {vector_store_path}")
        print(f"📊 Statistik:")
        print(f"   - Total chunks: {len(texts)}")
        print(f"   - Documents in vector store: {document_count}")
        print(f"   - Vector store size: {get_folder_size(vector_store_path):.1f} MB")
        print(f"   - Build time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return True

    except Exception as e:
        print(f"❌ Gagal membuat vector store: {e}")
        import traceback
        print(f"Error details: {traceback.format_exc()}")
        return False

def get_folder_size(folder_path):
    """Calculate folder size in MB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)  # Convert to MB

def main():
    parser = argparse.ArgumentParser(description='Pre-build vector store for RAG chatbot')
    parser.add_argument('--force', action='store_true',
                       help='Force rebuild even if vector store is up-to-date')
    parser.add_argument('--check', action='store_true',
                       help='Only check if rebuild is needed without building')

    args = parser.parse_args()

    print("🏥 Rare Disease Helper Chatbot - Vector Store Builder")
    print("=" * 60)

    if args.check:
        needs_rebuild, reason = should_rebuild_vectorstore("chroma_db")
        if needs_rebuild:
            print(f"🔄 Rebuild diperlukan: {reason}")
            sys.exit(1)
        else:
            print(f"✅ Vector store up-to-date: {reason}")
            sys.exit(0)

    # Set up environment (you might need to set GOOGLE_API_KEY)
    if 'GOOGLE_API_KEY' not in os.environ:
        print("⚠️ Warning: GOOGLE_API_KEY environment variable not set")
        print("Make sure to set your Google AI API key before running this script")

        # Try to read from .env file if available
        if os.path.exists('.env'):
            try:
                with open('.env', 'r') as f:
                    for line in f:
                        if line.startswith('GOOGLE_API_KEY='):
                            key = line.split('=', 1)[1].strip().strip('"')
                            os.environ['GOOGLE_API_KEY'] = key
                            print("✅ API key loaded from .env file")
                            break
            except:
                pass

        if 'GOOGLE_API_KEY' not in os.environ:
            print("❌ Please set GOOGLE_API_KEY environment variable")
            sys.exit(1)

    # Build vector store
    success = build_vector_store(force_rebuild=args.force)

    if success:
        print("\n🎉 Vector store berhasil dibuat!")
        print("Sekarang aplikasi Streamlit akan load lebih cepat.")
    else:
        print("\n❌ Gagal membuat vector store")
        sys.exit(1)

if __name__ == "__main__":
    main()
