#!/usr/bin/env python3
"""
Simple test script to check if vector store can be loaded
"""

import os
import glob
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

print("🧪 Testing Vector Store Loading...")

# Check environment
if not os.getenv('GOOGLE_API_KEY'):
    print("❌ GOOGLE_API_KEY not found in environment")
    exit(1)

# Check documents
md_files = glob.glob('./data/**/*.md', recursive=True)
print(f"📄 Found {len(md_files)} markdown files:")
for file in md_files:
    print(f"  - {file}")

if not md_files:
    print("❌ No markdown files found in data/ directory")
    exit(1)

# Try to load documents
print("\n📖 Loading documents...")
all_docs = []
for file_path in md_files:
    try:
        loader = UnstructuredMarkdownLoader(file_path)
        docs = loader.load()
        all_docs.extend(docs)
        print(f"  ✅ Loaded {len(docs)} chunks from {os.path.basename(file_path)}")
    except Exception as e:
        print(f"  ❌ Failed to load {file_path}: {e}")

if not all_docs:
    print("❌ No documents loaded")
    exit(1)

print(f"\n📝 Total documents loaded: {len(all_docs)}")

# Try to split documents
print("\n✂️ Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(all_docs)
print(f"✅ Created {len(texts)} text chunks")

# Try to create embeddings
print("\n🧠 Creating embeddings...")
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    print("✅ Embeddings model initialized")
except Exception as e:
    print(f"❌ Failed to initialize embeddings: {e}")
    exit(1)

# Try to create vector store
print("\n💾 Creating vector store...")
try:
    if os.path.exists("chroma_db"):
        print("🗑️ Removing existing vector store...")
        import shutil
        shutil.rmtree("chroma_db")

    vector_store = Chroma.from_documents(texts, embeddings, persist_directory="chroma_db")
    print("✅ Vector store created successfully!")

    # Test retrieval
    print("\n🔍 Testing retrieval...")
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    test_docs = retriever.get_relevant_documents("penyakit langka")
    print(f"✅ Retrieved {len(test_docs)} relevant documents for test query")

    print("\n🎉 All tests passed! Vector store is working correctly.")

except Exception as e:
    print(f"❌ Failed to create vector store: {e}")
    exit(1)
