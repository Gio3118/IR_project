from flask import Flask, request, make_response
import requests
import os
import pandas as pd

from SentimentAnalyzer import BERTSentimentAnalyzer


SENTIMENTS = {
    0: "Extremely Negative",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Extremely Positive",
}


def sentiment_to_str(sentiment: int) -> str:
    return SENTIMENTS.get(sentiment, "None")


app = Flask(__name__)
_DOCKER = os.environ.get("DOCKER", False)

DATA_FOLDER = (
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "data"))
    if not _DOCKER
    else "/data"
)


# @app.before_first_request
# def initModel():
#     try:
path_cur_file = os.path.dirname(__file__)
model_path = os.path.join(path_cur_file, "models/BertModel-acc87.pt")
model = BERTSentimentAnalyzer(model_path=model_path, gpu_enabled=False)
# session["model_loaded"] = True
# except:
#     session["model_loaded"] = False


@app.route("/sentiments", methods=["POST"])
def get_sentiments():
    if not model:
        return {
            "message": "Analyzing sentiments failed, model could not be loaded."
        }, 503
    if not os.path.exists(os.path.join(DATA_FOLDER, "scraper-staging", "staging.json")):
        return {"message": "No tweets to analyze."}, 400
    tweets = pd.read_json(os.path.join(DATA_FOLDER, "scraper-staging", "staging.json"))
    sentiments = model.analyze_batch(list(tweets.tweet.values))
    tweets["sentiment"] = sentiments
    tweets["sentiment"] = tweets["sentiment"].apply(sentiment_to_str)
    tweets.to_json(
        os.path.join(DATA_FOLDER, "sentiments-staging", "staging.json"),
        orient="records",
    )
    return {"message": "Tweets have been analyzed."}, 201
