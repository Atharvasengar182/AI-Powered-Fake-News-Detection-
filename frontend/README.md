# AI-Powered Fake News Detector - Frontend

This is the Streamlit-based web frontend for the AI-Powered Fake News Detector. It connects to the FastAPI backend service via HTTP POST requests and visualizes predictions, confidence scores, probability breakdowns, and classification limitations.

## Prerequisites

Ensure the backend server is running first from the `backend/` directory:

```bash
cd backend
source .venv/bin/activate
uvicorn app:app --reload
```
The backend API runs at `http://127.0.0.1:8000`.

## Frontend Setup

1. Open a new terminal and navigate to the `frontend/` directory:
   ```bash
   cd frontend
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

The app will open automatically in your browser at `http://localhost:8501`.

## Features

- **Text Classification**: Paste news headlines or full articles (up to 100,000 characters).
- **Probabilities & Confidence**: View confidence percentage and a side-by-side probability breakdown for Real vs. Fake news.
- **Pre-loaded Samples**: One-click selection of sample news articles in the sidebar for quick testing.
- **Configurable Backend**: Backend URL can be set via the `BACKEND_URL` environment variable or adjusted dynamically in the sidebar.
- **Responsive Layout**: Designed for seamless usability on desktop and mobile screens.
- **Explicit Disclaimers**: Clarifies that predictions reflect statistical linguistic pattern resemblance to the WELFake corpus, not independent fact verification.
