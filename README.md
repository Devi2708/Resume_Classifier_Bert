# Resume Classifier using BERT

This project classifies resumes into job roles (e.g., Data Scientist, Analyst) using a fine-tuned BERT model.

## Features
- Text preprocessing with Transformers Tokenizer
- Fine-tuned BERT with HuggingFace
- Streamlit dashboard to upload resumes and show predictions
- SHAP explainability and MLflow tracking

## Quick start
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```