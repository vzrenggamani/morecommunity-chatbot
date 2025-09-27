# 🏥 Rare Disease Helper Chatbot

A simple AI chatbot to help with rare disease information using Google Gemini and RAG (Retrieval-Augmented Generation).

## 🚀 Quick Start

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Add Your API Key
Create a `.env` file:
```
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Run the App
```bash
# Windows
start.bat

# Linux/Mac
chmod +x start.sh && ./start.sh
```

### 4. Open Browser
Go to: **http://localhost:8501**

## 🐳 Docker (Optional)
```bash
docker-compose -f docker-compose.simple.yml up
```

## 📁 Add Your Documents
Put your markdown documents in the `data/` folder:
```
data/
├── medical_reference/
├── user_stories/
└── community_resources/
```

## 📋 Features
- Chat with AI about rare diseases
- Document-based knowledge retrieval
- Empathetic responses based on real experiences
- Auto-builds knowledge base from your documents

That's it! 🎉
