# Rare Disease Helper Chatbot - Docker Deployment

This guide explains how to deploy the Rare Disease Helper Chatbot using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose (usually included with Docker Desktop)
- Google Gemini API key

## Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Set your Google API Key:**
   ```bash
   # On Windows (PowerShell)
   $env:GOOGLE_API_KEY="your_google_api_key_here"

   # On Linux/macOS
   export GOOGLE_API_KEY="your_google_api_key_here"
   ```

2. **Build and run the container:**
   ```bash
   docker-compose up --build
   ```

3. **Access the application:**
   Open your browser and go to: `http://localhost:8501`

### Option 2: Using Docker directly

1. **Build the Docker image:**
   ```bash
   docker build -t raredisease-chatbot .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8501:8501 -e GOOGLE_API_KEY="your_google_api_key_here" -v "./data:/app/data:ro" raredisease-chatbot
   ```

3. **Access the application:**
   Open your browser and go to: `http://localhost:8501`

## Environment Variables

- `GOOGLE_API_KEY`: Your Google Gemini API key (required)

## Volumes

- `./data:/app/data:ro`: Mounts your local data directory (read-only)
- `chroma_data:/app/chroma_db`: Persists ChromaDB vector database

## Stopping the Application

### If using Docker Compose:
```bash
docker-compose down
```

### If using Docker directly:
```bash
# Find the container ID
docker ps

# Stop the container
docker stop <container_id>
```

## Development

For development purposes, you can mount the entire project directory:

```bash
docker run -p 8501:8501 \
  -e GOOGLE_API_KEY="your_api_key" \
  -v "$(pwd):/app" \
  -w /app \
  python:3.11-slim \
  bash -c "pip install -r requirements.txt && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"
```

## Troubleshooting

### Common Issues:

1. **Port already in use:**
   - Change the port mapping: `-p 8502:8501` (then access via `http://localhost:8502`)

2. **API key not working:**
   - Ensure your Google API key is valid and has access to Gemini API
   - Check that the environment variable is properly set

3. **No documents loaded:**
   - Ensure the `data` directory contains markdown files
   - Check that the volume mount is correct

4. **ChromaDB issues:**
   - Clear the ChromaDB volume: `docker-compose down -v`
   - Rebuild the container: `docker-compose up --build`

### Viewing Logs:

```bash
# With Docker Compose
docker-compose logs -f

# With Docker directly
docker logs <container_id> -f
```
