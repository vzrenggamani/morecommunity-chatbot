# Build and tag

docker build -t ghcr.io/your-username/raredisease-chatbot:latest .

# Login to GitHub Container Registry

echo $GITHUB_TOKEN | docker login ghcr.io -u your-username --password-stdin

# Push

docker push ghcr.io/your-username/raredisease-chatbot:latest
