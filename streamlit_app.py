import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import shap
import torch

# Load model & tokenizer
model_path = "models/resume_classifier"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.eval()

st.title("Resume Classifier with SHAP Explainability")

uploaded_file = st.file_uploader("Upload Resume (.txt)", type=["txt"])
if uploaded_file:
    resume_text = uploaded_file.read().decode("utf-8", errors="ignore")

    st.subheader("Resume Preview")
    st.text(resume_text[:1000])

    # Tokenize input
    inputs = tokenizer(resume_text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_label = torch.argmax(probs).item()

    role_map = {0: "Data Analyst", 1: "Data Scientist", 2: "Backend Developer"}
    st.success(f"Predicted Role: {role_map[pred_label]}")

    # SHAP Explainability
    st.subheader("Model Explanation (SHAP)")
    explainer = shap.Explainer(model, tokenizer)
    shap_values = explainer([resume_text])

    # Render HTML SHAP output
    st.components.v1.html(shap.plots.text(shap_values[0], display=False), height=400, scrolling=True)
