# Product Review Analyzer

A simple scaffold for analyzing product reviews: preprocessing, training a sentiment model, predicting, keyword extraction, and basic recommendations.

Structure:
- `data/` — raw and processed datasets
- `models/` — trained model and vectorizer
- `src/` — core modules
- `app/` — Streamlit app
- `utils/` — helpers

Quick start:
1. Create a virtualenv and install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare data in `data/raw/` (CSV with `review` and `label` columns).
3. Train a model: `python src/train_model.py`
4. Run the app: `streamlit run app/app.py`
