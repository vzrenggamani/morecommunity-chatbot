# 1. Create project directory

mkdir raredisease-chatbot && cd raredisease-chatbot

# 2. Upload docker-compose.prod.yml and .env

scp docker-compose.prod.yml user@your-vps:/path/to/project/
scp .env user@your-vps:/path/to/project/

# 3. Run the application

docker-compose -f docker-compose.prod.yml up -d
