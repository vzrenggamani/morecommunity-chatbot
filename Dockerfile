# Simple Dockerfile for Rare Disease Helper Chatbot
FROM python:3.11-slim-trixie

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Create directories
RUN mkdir -p chroma_db logs

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Make startup script executable
RUN chmod +x /app/start.sh

# Run startup script
CMD ["/app/start.sh"]
