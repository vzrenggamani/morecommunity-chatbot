# Rare Disease Helper Chatbot - AI Development Guide

## Architecture Overview

This is a **Streamlit-based RAG (Retrieval-Augmented Generation) chatbot** for rare disease information, using Google AI/Gemini and ChromaDB. The application was recently refactored from a monolithic structure into modular components.

### Core Components

- **`app.py`** - Main entry point (47 lines), handles routing between chat and debug pages
- **`pages/`** - Streamlit page components (`chat_page.py`, `debug_page.py`)
- **`utils/`** - Business logic modules (`llm_utils.py`, `token_tracking.py`, `document_utils.py`, `vectorstore_utils.py`)
- **`data/`** - Document knowledge base with typed folders (`medical_reference/`, `user_stories/`, `community_resources/`)

### Critical Architectural Patterns

**1. ChromaDB SQLite Compatibility Patch**

```python
# ALWAYS include this at the top of files using ChromaDB
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
```

**2. Document Type Classification**
Documents are automatically categorized by folder structure in `utils/document_utils.py`:

- `medical_reference/` → `medical_reference`
- `user_stories/` → `user_story`
- `community_resources/` → `community_resource`

**3. Smart Vector Store Caching**
The app uses `@st.cache_resource` and timestamp-based rebuilding in `utils/llm_utils.py`. Vector store rebuilds only when source documents are newer than the existing `chroma_db/`.

## Development Workflows

### Local Development

```bash
# Setup environment
cp .env.example .env
# Edit .env with GOOGLE_API_KEY

# Run with Docker (recommended)
docker-compose up -d --build

# Access at http://localhost:8501
```

### Vector Store Management

```bash
# Pre-build vector store (eliminates cold starts)
python build_vector_store.py

# Force rebuild
python build_vector_store.py --force

# Check if rebuild needed
python build_vector_store.py --check
```

### Ubuntu VPS Deployment

```bash
# On VPS - Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Deploy pre-built image (recommended for cost optimization)
docker pull your-registry/raredisease-chatbot:latest
docker-compose -f docker-compose.prod.yml up -d

# Or build locally and save image
docker save raredisease-chatbot:latest | gzip > chatbot.tar.gz
scp chatbot.tar.gz user@vps:/tmp/
# On VPS: gunzip -c /tmp/chatbot.tar.gz | docker load
```

### Debugging Workflows

- Use **Debug Page** (`🔧 Debug Info`) for vector store inspection
- **"🔍 Show Full Prompt"** buttons reveal exact LLM inputs including RAG context
- **"🔍 Debug Response Structure"** shows Google AI API response metadata

### Cost Optimization Strategies

- **Pre-build vector stores** to avoid repeated embedding API calls
- **Monitor token usage** in real-time via sidebar metrics
- **Optimize context window**: Current retrieval limited to k=3 documents, score_threshold=0.3
- **Cache LLM responses**: Use `@st.cache_resource` for repeated queries
- **Batch document processing**: Use `build_vector_store.py` for bulk operations

## Key Conventions

### Document Processing

- Markdown files in `data/` subfolders are auto-processed
- Text chunking: 1000 chars with 100 char overlap
- Document metadata includes `document_type`, `source_category`, `original_source`

### Token Management

- **Hybrid counting**: Manual estimation using `tiktoken` with Gemini correction factors + API metadata extraction
- **Real-time tracking**: Sidebar displays input/output tokens with cost estimation
- **Session analytics**: Tracks conversation history and cumulative usage
- **Cost optimization**:
  ```python
  # Token limits enforced in utils/llm_utils.py
  max_output_tokens=1024  # Configurable limit
  k=3  # Reduced retrieval documents (was 5)
  score_threshold=0.3  # Filters low-relevance context
  ```

### Medical Domain Prompt Engineering

The system uses sophisticated prompt engineering for Indonesian medical consultations:

**Context Integration Strategy:**

```python
# Prompt structure in utils/llm_utils.py
custom_prompt_template = """
Anda adalah dokter AI yang ahli dalam bidang penyakit langka dan kesehatan.

KONTEKS UMUM (bukan data pasien saat ini):
{context}

PERTANYAAN PASIEN: {question}

PENTING - JANGAN LAKUKAN:
- Informasi dalam konteks hanya sebagai referensi umum, BUKAN bagian dari riwayat pasien
- JANGAN mengasumsikan kondisi atau gejala pasien kecuali dinyatakan dalam pertanyaan
- JANGAN rujuk ke sumber atau dokumen secara eksplisit

LAKUKAN:
- Pisahkan antara penjelasan medis umum dan rekomendasi langkah praktis
- Berikan panduan umum untuk proses evaluasi medis
- Selalu sarankan konsultasi medis profesional untuk diagnosis dan pengobatan
"""
```

**Safety & Compliance Approach:**

- **Medical disclaimers**: Every response includes professional consultation recommendations
- **Source separation**: Clear distinction between general medical knowledge and patient-specific advice
- **Cultural adaptation**: Indonesian language with appropriate medical terminology
- **Empathy integration**: Uses patient stories for emotional support without assuming patient conditions

### Vector Store Optimization

**Cost-Effective Strategies:**

```python
# Document chunking optimization (utils/llm_utils.py)
chunk_size=1000      # Balanced between context and cost
chunk_overlap=100    # Minimal overlap to reduce redundancy

# Embedding efficiency (build_vector_store.py)
GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")  # Cost-effective model
```

**Smart Rebuilding Logic:**

- **Timestamp comparison**: Only rebuilds when source documents are newer
- **Incremental updates**: Considers adding differential updates for large document sets
- **Persistence strategy**: ChromaDB volume mounting preserves embeddings across deployments

### UI Patterns

- Indonesian language interface (`"Apa yang ingin Anda tanyakan?"`)
- Emoji-based status indicators (`✅`, `⚠️`, `❌`, `🔄`)
- Source attribution with document type mapping:
  ```python
  type_display = {
      "medical_reference": "📚 Referensi Medis",
      "user_story": "👥 Pengalaman Keluarga",
      "community_resource": "🏘️ Sumber Komunitas"
  }
  ```

### LLM Configuration

- Model: `gemini-2.0-flash-lite`
- Custom prompt engineering for medical domain in Indonesian
- Retrieval: similarity search with score threshold 0.3, k=3 documents
- Temperature: 0.7, max_output_tokens: 1024

## Critical Integration Points

### Google AI Dependencies

- **Embeddings**: `GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")`
- **Chat**: `ChatGoogleGenerativeAI` with specific safety settings
- **Environment**: Requires `GOOGLE_API_KEY` in `.env`

### Docker Environment Handling

- Data directory detection supports multiple paths: `./data`, `/app/data`, `data`
- ChromaDB persistence via Docker volume `chroma_data:/app/chroma_db`
- Health check endpoint: `http://localhost:8501/_stcore/health`

### Streamlit State Management

- Session state keys: `messages`, `token_usage`, `prompt_history`
- Cache invalidation handled by `@st.cache_resource` decorator
- Page routing via `st.sidebar.selectbox`

## Deployment Notes

- **Production**: Use `docker-compose.prod.yml` with pre-built images
- **Vector Store**: Pre-build with `build_vector_store.py` to avoid cold starts
- **Environment**: Mount data as read-only, persist only ChromaDB volume
- **Monitoring**: Built-in token tracking and debug endpoints
- **VPS Deployment**: Optimized for Ubuntu servers (EC2/GCP/Baremetal) with Docker
- **Cost Optimization**: Pre-built images reduce deployment time and API calls

## Cost Management Patterns

### Token Usage Optimization

```python
# In pages/chat_page.py - Monitor and limit context
def optimize_context_length(retrieved_docs, max_tokens=500):
    """Truncate context to fit token budget"""
    # Implementation prioritizes most relevant docs first
    # Stops adding context when token limit approached
```

### Embedding Cost Reduction

```bash
# Build vector store once, reuse across deployments
docker run --rm -v $(pwd)/data:/app/data -v chroma_data:/app/chroma_db \
  your-image python build_vector_store.py

# Deploy with pre-built embeddings
docker-compose -f docker-compose.prod.yml up -d
```

### API Call Monitoring

- **Sidebar metrics**: Real-time token count and cost estimation
- **Session tracking**: Cumulative usage across conversations
- **Debug tools**: Inspect exact prompts sent to API for optimization opportunities

## When Making Changes

1. **Adding documents**: Place in appropriate `data/` subfolder, vector store auto-rebuilds
2. **UI changes**: Modify page components in `pages/`, maintain Indonesian language patterns
3. **LLM changes**: Update `utils/llm_utils.py`, consider token limit implications
4. **New utilities**: Add to `utils/` with proper imports in `__init__.py`
