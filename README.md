# 🛡️ TruthLens AI – Fake News Detection Framework

TruthLens AI is an end-to-end Machine Learning and NLP platform designed to detect disinformation and linguistic deception in news articles and headlines. Trained on the benchmark **WELFake Dataset (72,134 news items)**, the system achieves **95.26% test accuracy** using TF-IDF N-Gram Vectorization and L2-Regularized Logistic Regression.

---

## 🌟 Key Features

- **🏠 Curated Disinformation Showcase**: Explore famous viral fake news stories that were flagged and debunked by the AI (e.g., Pope Francis endorsement hoax, NASA 15-day darkness hoax).
- **🔬 Sentence-Level Deception Heat-Map**: Unlike black-box detectors, TruthLens evaluates articles sentence-by-sentence to show **what percentage and which specific parts** of an article contain deceptive linguistic signals.
- **🎨 Interactive Highlighting**: Real-time visual color-coding:
  - 🔴 **Red (>65%)**: High-risk / sensationalist claims
  - 🟡 **Yellow (40–65%)**: Moderate / unverified markers
  - 🟢 **Green (<40%)**: Credible / objective journalistic style
- **🔑 Key Vocabulary Attribution**: Dynamically extracts top contributing vocabulary tokens pushing scores towards Real vs. Fake news.
- **⚡ Fast Inference**: Sub-50ms inference latency with fully cached model artifacts.

---

## 📊 Model Benchmarks

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **95.26%** |
| **Precision** | **95.10%** |
| **Recall** | **95.40%** |
| **F1-Score** | **95.25%** |
| **Training Corpus** | 72,134 Articles (WELFake) |

---

## 🚀 Quickstart (Run Locally)

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/fake-news-detector.git
cd fake-news-detector
```

### 2. Set up Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Launch Streamlit Web Dashboard
```bash
streamlit run frontend/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🌐 Deploying to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io).
3. Click **"New app"**.
4. Select your repository and set:
   - **Branch:** `main`
   - **Main file path:** `frontend/app.py`
5. Click **"Deploy"**!

---

## ⚖️ Ethical Disclaimer
TruthLens AI generates statistical linguistic probability estimates based on vocabulary resemblance to historical training corpora. It is designed as an investigative assistant and does not replace primary journalistic fact-checking.
