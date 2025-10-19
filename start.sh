#!/bin/bash

# Check if running in Docker container
if [ -f /.dockerenv ]; then
    echo "Starting Rare Disease Helper Chatbot in Docker..."
    echo "Working directory: $(pwd)"
    echo "Directory contents: $(ls -la)"
    echo "Data directory contents: $(ls -la data/ 2>/dev/null || echo 'Data directory not found')"

    # Check environment variables
    echo "Environment check:"
    echo "GOOGLE_API_KEY: ${GOOGLE_API_KEY:+Set} ${GOOGLE_API_KEY:-Not set}"

    # Try to build vector store, but don't fail if it errors
    echo "Attempting to build vector store..."
    if python build_vector_store.py --force; then
        echo "Vector store built successfully"
    else
        echo "Vector store build failed, but continuing with app startup"
        echo "Vector store will be built on first use in the app"
    fi

    echo "Starting Streamlit app..."
    exec streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true

    else
        echo "Starting Simple Development Server..."

        # Check if .env exists
        if [ ! -f ".env" ]; then
            echo "Creating .env file..."
            echo "GOOGLE_API_KEY=your_api_key_here" > .env
            echo "Please edit .env and add your Google API key, then run this script again."
            exit 1
        fi

        # Start Streamlit
        echo "Starting on http://localhost:8501"
        streamlit run app.py
    fi
