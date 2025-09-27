@echo off
echo 🚀 Starting Simple Development Server...

REM Check if .env exists
if not exist ".env" (
    echo Creating .env file...
    echo GOOGLE_API_KEY=your_api_key_here > .env
    echo Please edit .env and add your Google API key, then run this script again.
    pause
    exit /b 1
)

REM Start Streamlit
echo Starting on http://localhost:8501
streamlit run app.py
