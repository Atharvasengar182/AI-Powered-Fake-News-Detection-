# Frontend handoff: AI-Powered Fake News Detector

Build the frontend for this existing project. The backend source and trained model are in the `backend/` folder.

## Project goal

Create a simple interface where a user can paste a news headline or article, submit it for classification, and see whether the text resembles the WELFake dataset's Fake News or Real News class. This is a machine-learning prediction, not independent fact-checking. Make that limitation clear in the interface.

## Recommended implementation

Use Streamlit, as specified in the project workflow. Keep the frontend in a separate `frontend/` folder and call the API from the Streamlit server with an HTTP POST request. Do not load the model artifact in the frontend.

The API runs at `http://127.0.0.1:8000` during local development. Its interactive documentation is at `http://127.0.0.1:8000/docs`.

### Request

`POST http://127.0.0.1:8000/predict`

```json
{
  "text": "A headline or full news article"
}
```

### Successful response

```json
{
  "prediction": "FAKE NEWS",
  "confidence": 0.91,
  "probabilities": {
    "FAKE NEWS": 0.91,
    "REAL NEWS": 0.09
  },
  "disclaimer": "This is a model prediction, not verified proof that the story is true or false."
}
```

`confidence` and each probability are decimals from 0 to 1. Show confidence as a percentage in the UI. The response may predict either `FAKE NEWS` or `REAL NEWS`.

## Interface requirements

- Clear title and a short explanation of what the detector does.
- Large text area for a headline or article.
- Predict button with a loading state while the API request runs.
- Validate empty input before sending it.
- Show the predicted class, confidence percentage, and both class probabilities when present.
- Display the response disclaimer near the result; do not describe confidence as the probability that the story is factually true.
- Show useful messages when the backend is unavailable or returns an error. The API returns HTTP 422 for empty text and HTTP 413 for text longer than 100,000 characters.
- Include a clear/reset action and make the layout usable on a narrow screen.

## Running locally

Start the backend in one terminal from the `backend/` folder:

```bash
.venv/bin/python -m uvicorn app:app --reload
```

Start the frontend in a second terminal using the instructions for the chosen Streamlit setup. Keep both processes running. The frontend should call the backend URL through one configurable setting, defaulting to `http://127.0.0.1:8000`.

## Delivery

Implement the frontend files, add a frontend README with setup/run commands, and keep the backend API contract above intact. Do not retrain or replace the classifier as part of the frontend task.
