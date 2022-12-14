from flask import Flask, request, make_response
import requests
import os

from SentimentAnalyzer import BERTSentimentAnalyzer


app = Flask(__name__)

model_loaded = False
try:
    path_cur_file = os.path.dirname(__file__)
    model_path = os.path.join(path_cur_file, "models/BertModel-acc87.pt")
    model = BERTSentimentAnalyzer(model_path=model_path)
    model_loaded = True
except:
    model_loaded = False


@app.route("/sentiments", methods=["POST"])
def get_sentiments():
    if not model_loaded:
        return {
            "message": "Analyzing sentiments failed, model could not be loaded."
        }, 503
    sentiments = model.analyze_batch(request.json["tweets"])
    return {"message": "Tweets analyzed correctly.", "sentiments": sentiments}, 401
