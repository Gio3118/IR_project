from flask import Flask, request, make_response, render_template
import requests
import os

DIST_DIR = os.path.join(os.path.dirname(__file__), "dist")
STATIC_DIR = os.path.join(DIST_DIR, "assets")
app = Flask(__name__, template_folder=DIST_DIR, static_folder=STATIC_DIR)

_DOCKER = os.environ.get("DOCKER", False)
DATA_FOLDER = (
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "data"))
    if not _DOCKER
    else "/data"
)

scraper_uri = "http://scraping-service:5000" if _DOCKER else "http://127.0.0.1:5001"
sentiment_service_uri = (
    "http://sentiment-analysis-service:5000" if _DOCKER else "http://127.0.0.1:5002"
)
search_service_uri = "http://search-service:5000" if _DOCKER else None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search", methods=["GET"])
def scrape():
    query = request.args.get("query")
    selector = request.args.get("selector")

    if query is None or query == "":
        return make_response("No query provided", 400)

    if selector == "scrape":
        limit = request.args.get("limit", 100)
        response = requests.get(
            f"{scraper_uri}/tweets", params={"query": query, "limit": limit}
        )
        if not response.ok:
            return make_response("Scraping failed", 400)

        # Sentiment analysis
        response = requests.post(f"{sentiment_service_uri}/sentiments")
        os.remove(os.path.join(DATA_FOLDER, "scraper-staging", "staging.json"))
        if not response.ok:
            return make_response("Sentiment analysis failed", 400)
        if search_service_uri:
            response = requests.post(f"{search_service_uri}/index")
            os.remove(os.path.join(DATA_FOLDER, "sentiments-staging", "staging.json"))
            if not response.ok:
                return make_response("Failed to index the new data.", 400)
        else:
            return make_response("Search service not available.", 400)

        return {"message": "New tweets fetched, analyzed and indexed."}, 201
    elif selector == "fetch":
        page = 0
        if search_service_uri:
            response = requests.get(
                f"{search_service_uri}/search", params={"query": query, "page": page}
            )
            if not response.ok:
                return make_response("Search failed", 400)
            print(response)
            return response.json(), 200

    return {"message": f"Ok! {query} {selector} {search_service_uri}"}, 201
