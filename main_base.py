from flask import Flask
from utils import predict_tags

app = Flask(__name__)


@app.route('/')
def root():
    return 'Bienvenu sur cette API qui prédit les tags de vos questions Stackoverflow'


@app.route('/predict_tags/<string:question>')
def api_predict_tags(question):
    return predict_tags(question)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8999)