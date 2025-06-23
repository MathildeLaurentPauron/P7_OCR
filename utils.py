import tensorflow_hub as hub
import tensorflow
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import nltk 
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt_tab')

# Fonction de nettoyage de texte
def clean_text(text):
    text = re.sub(r'[^\w\s#+]', '', text)  # Enlever la ponctuation
    text = text.lower()  # Convertir en minuscules
    text = re.sub(r'<[^>]+>', '', text)  # Enlever les balises HTML
    return text

# Téléchargement et chargement du binariseur depuis MLflow
binarizer_path = "mlb_use.pkl"
with open(binarizer_path, "rb") as f:
    mlb = pickle.load(f)

# Charger le modèle USE depuis TensorFlow Hub
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Fonction pour obtenir l'embedding d'un texte avec USE
def get_use_embedding(question):
    return use_model([clean_text(question)]).numpy().flatten()

# Charger le modèle
model_path = "model_use.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

def predict_tags(question):
    embedding = get_use_embedding(question)
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    keywords = model.predict(embedding)
    predicted_tags = mlb.inverse_transform(keywords)
    return predicted_tags
    