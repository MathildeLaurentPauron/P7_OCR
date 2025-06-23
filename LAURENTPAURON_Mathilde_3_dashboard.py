# app_tags_predictor.py

import streamlit as st
import requests
from sklearn.metrics import jaccard_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:8b"

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
    tag_list = ', '.join(f'"{tag}"' for tag in tag_vocab)
    prompt = f"""
You are a StackOverflow tags predictor.

Your task is to output a short list of the most relevant tags for a programming question.

Constraints:
- Use only tags from this list: [{tag_list}]
- Select between 1 and 5 tags
- Do NOT return JSON, do NOT return numbered lists, do NOT use brackets or quotes
- Your output MUST be a simple, comma-separated list of tags, all in lowercase

### Examples:

Example 1:
Question:
ansible dockercompose module different result from ansible than over ssh on the host pi have a task in my ansible bitbucket role that simply starts two dockercomposeyml files\nwhen executing this from ansible it fails but when i execute the same command over ssh on the 
actual server it works just fine i am currently executing my playbook as root and when i manually execute dockercompose f ltpathgt up d i am also rootp\nprecode name create and start services\n name docker compose up\n docker_compose\n project_src quot 
bitbucket__install_path quot\n files quot bitbucket__compose_files quot\n state present\n register output\ncodepre\npansible commandp\nblockquote\npansibleplaybook i inventorystaging playbooksbitbucketserveryml vvvv kku root askvaultpass startattask quotsetup 
bitbucketquotp\nblockquote\npansible sucessfully starts 2 out of 3 containers but the third crashes with the errorp\nblockquote\nperror response from daemon oci runtime create failed\ncontainer_linuxgo380 starting container process caused exec\nquotmy_entrypointshquot 
permission denied unknown error failed to start\ncontainers nginx_bitbucket_1p\nblockquote\npmy question is why do ansible receive an error when executing the above task while executing quotdockercompose f ltpathgt up d over ssh works just fine what is the difference 
between the two casesp

Tags:
python

Example 2:
Question:
function returns address of local variable wreturnlocaladdr sprintf pi ma new c and i am trying sprintf along with pointers all i get in console is return buf as is please help me with this codep\nprecode#include ltstdiohgt\nchar stringachar str\nint main\n\n char ss 
quot123quot\n stringass\n\n return 0\n\n\nchar stringa char str\n\n char buf 100 \n sprintfbufquothello squot str\n return buf\n\n \ncodepre\npi tried many other ways too like sprintf_c and my computer shut down for serious i am learning cp

Tags:
arrays

Example 3:
Question:
removing a drag item from an angular materials cdkdroplist when the element is dragged far away from the container pi have a cdkdroplist with 4 draggable items inside the problem is that i want to be able to drag an item completely out of the drop list once i drag it 
far away from its container currently it doesnt matter how far you drag the item the item still returns within the boundary of the drop listp\npas it is shown in the documentation of angular materials a hrefhttpsmaterialangulariocdkdragdropoverview#cdkdragdropconnectedsorting 
relnofollow noreferrerlink to the examplea it is possible to move items within the drop list and fromto a drop list to anotherp\npi want to know if the behavior that im looking for even possiblep\npin short i want to be able to freely move the item in the page without 
the item returning back to the drop list container once i release the mouse button in other words to detach the item from the drop listp

Tags:
html, css

Example 4:
Question:
is there anyway to list existing routes in feathersjs framework pin laravel i can usep\npcodephp artisan route listcodep\npto see all routes in my stronglaravelstrong project i am asking for the emsame behaviorem but in strongfeathersjsstrong framework which built on top 
of expressjsp\npif not is there anyway to create that as a custom command using node js to obtain the same behavior p\npthanksp

Tags:
javascript, node.js

Example 5:
Question:
is there a google sheets function that allow you to do multiple queries for different data sets with similar fields but different conditions pso i have a couple of google sheets tabs that contain similar data and want to merge into a master sheet but have two different 
conditions for each query i have triedp\nprecodequeryinventoryazquotselect col1 col2 col3 col4 col5 col15 col17 col18 col7 col11 col12 col13 col14 where col15 is not null and col15 lt quotampf1ampquot and col19 ltgt printed and col20 restock order by col15 asc limit 
200quot1querypreinventoryaoquotselect col1 col2 col3 col4 col6 col7 col8 col9 col10 col11 col12 col13 col14 where col15 xquot\ncodepre\npandp\nprecodequeryimportrangequoturlquot quotinventoryazquotquotselect col1 col2 col3 col4 col5 col15 col17 col18 col7 col11 col12 
col13 col14 where col15 is not null and col15 lt quotampf1ampquot and col19 ltgt printed and col20 restock order by col15 asc limit 200quot1queryimportrangequoturlquot quotpreinventorya3p2000quotquotselect col1 col2 col3 col4 col5 col7 col8 col9 col10 col11 col12 col13 
col14 where col16 xquot\ncodepre

Tags:
arrays


Question:
---
{question_text}
---

Tags:
"""

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    })

    result = response.json()
    output_text = result.get("response", "").lower().strip()

    # Nettoyage de la sortie pour éviter les formats indésirables (brackets, quotes, etc.)
    output_text = re.sub(r'[\[\]"\']', '', output_text)
    raw_tags = [tag.strip() for tag in output_text.split(',')]
    predicted_tags = list(dict.fromkeys([tag for tag in raw_tags if tag in tag_vocab]))[:5]
    
    return predicted_tags

def compute_jaccard(true_tags, predicted_tags):
    mlb = pickle.load(open("mlb_use.pkl", "rb"))
    y_true = mlb.transform([true_tags])
    y_pred = mlb.transform([predicted_tags])
    return jaccard_score(y_true, y_pred, average='samples')

def compute_f1score(true_tags, predicted_tags):
    mlb = pickle.load(open("mlb_use.pkl", "rb"))
    y_true = mlb.transform([true_tags])
    y_pred = mlb.transform([predicted_tags])
    return f1_score(y_true, y_pred, average='macro')

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
        score_jacc = compute_jaccard(true_tags, predicted_tags)
        score_f1 = compute_f1score(true_tags, predicted_tags)
        st.subheader("📊 Jaccard Score")
        st.write(f"{score_jacc:.4f}")
        st.subheader("📊 F1 Score")
        st.write(f"{score_f1:.4f}")
    else:
        st.info("💡 Entrez les tags attendus (séparés par des virgules) pour voir le Jaccard score.")