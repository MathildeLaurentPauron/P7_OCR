# app_tags_predictor.py

import streamlit as st
import requests
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:1.5b"

# Liste fixe de tags possibles
TAG_VOCAB = [
    "python",
    "javascript",
    "java",
    "reactjs",
    "html",
    "node.js",
    "android",
    "pandas",
    "css",
    "arrays" 
]

def ask_ollama(question_text, tag_vocab):
    tag_list = ", ".join(tag_vocab)
    prompt = f"""
You are a StackOverflow tags predictor.

Your task is to select the most relevant tags for a question.

Instructions:
- Only use tags from this list: [{tag_list}]
- Select between 1 and 5 relevant tags
- Format: comma-separated list of tags
- No explanation, no numbering, no repetition

Question:
---
{question_text}
---

Respond with tags only:
"""

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    })

    result = response.json()
    output_text = result.get("response", "").lower().strip()
    raw_tags = output_text.lower().split(',')
    predicted_tags = list(dict.fromkeys([tag.strip() for tag in raw_tags if tag.strip() in tag_vocab]))[:5]
    return predicted_tags

def compute_jaccard(true_tags, predicted_tags, all_tags):
    mlb = MultiLabelBinarizer(classes=all_tags)
    y_true = mlb.fit_transform([true_tags])
    y_pred = mlb.transform([predicted_tags])
    return jaccard_score(y_true, y_pred, average='samples')

# Interface Streamlit
st.title("🧠 Prédiction de Tags StackOverflow par LLM (via Ollama)")

question_text = st.text_area("Entrez votre question StackOverflow :", height=300)
true_tags_input = st.text_input("(Optionnel) Tags attendus (séparés par des virgules)")

if st.button("Prédire les tags"):
    predicted_tags = ask_ollama(question_text, TAG_VOCAB)

    st.subheader("🏷️ Tags prédits")
    st.write(predicted_tags)

    if true_tags_input:
        true_tags = [tag.strip().lower() for tag in true_tags_input.split(",") if tag.strip()]
        score = compute_jaccard(true_tags, predicted_tags, TAG_VOCAB)
        st.subheader("📊 Jaccard Score")
        st.write(f"{score:.4f}")
    else:
        st.info("💡 Entrez les tags attendus (séparés par des virgules) pour voir le Jaccard score.")