from flask import Flask, make_response, request
import requests
import snscrape.modules.twitter as twitter
import pandas as pd
import os


app = Flask(__name__)

_DOCKER = os.environ.get("DOCKER", False)
DATA_FOLDER = (
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "data"))
    if not _DOCKER
    else "/data"
)


def get_tweets(query: str, limit: int = 5000) -> pd.DataFrame:
    holding = []

    for i, tweet in enumerate(
        twitter.TwitterSearchScraper(query + " lang:en").get_items()
    ):
        if i > limit:
            break
        holding.append([tweet.username, tweet.content])

    df = pd.DataFrame(holding, columns=["user", "tweet"])
    return df


@app.route("/tweets", methods=["GET"])
def get_tweets_route():
    query = request.args.get("query")
    limit = request.args.get("limit")
    if limit is not None:
        limit = int(limit)
    else:
        limit = 5000
    if query is None or query == "":
        return make_response("No query provided", 400)

    df = get_tweets(query, limit)
    df.to_json(
        os.path.join(DATA_FOLDER, "scraper-staging", "staging.json"), orient="records"
    )
    return {"message": "New tweets have been fetched."}, 201
