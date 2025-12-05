# Deploy this Streamlit app

Put these files in the same folder (or in a GitHub repo root):
- app.py (use app_updated.py but rename it to app.py)
- ACC_COMPLAINTS.csv
- PROCUREMENT_FRAUD.csv
- FRAUD_PATTERNS.json
- EVIDENCE_OCR_DATASET.txt
- requirements.txt

## Run locally
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501

## Deploy on Streamlit Community Cloud
1) Push the repo to GitHub (public or private).
2) Go to Streamlit Community Cloud, click "New app".
3) Choose the repo, branch, and set Main file path to `app.py`.
4) If you need a specific Python version, set it from the app's settings on the Cloud UI.

## Deploy on Hugging Face Spaces (Streamlit)
Create a new Space, choose the Streamlit SDK, push the same repo contents.

