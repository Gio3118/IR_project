from flask import Flask, request, make_response, render_template
import requests
import os

import lucene
from java.nio.file import Paths

from org.apache.lucene.store import NIOFSDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader


from searcher import run
from indexer import indexFile

app = Flask(__name__)

_DOCKER = os.environ.get("DOCKER", False)
DATA_FOLDER = (
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "data"))
    if not _DOCKER
    else "/data"
)

lucene.initVM(vmargs=["-Djava.awt.headless=true"])


@app.before_request
def attach_thread():
    lucene.getVMEnv().attachCurrentThread()


def init_searcher():
    try:
        path = Paths.get(os.path.join(os.path.dirname(__file__), "index"))
        directory = NIOFSDirectory(path)
        searcher = IndexSearcher(DirectoryReader.open(directory))
        analyzer = StandardAnalyzer()
        return searcher, analyzer
    except Exception as e:
        print("Error: ", e)
        return None, None


@app.route("/index", methods=["POST"])
def index_staged():
    staged_file_path = os.path.join(DATA_FOLDER, "sentiments-staging", "staging.json")
    if not os.path.exists(staged_file_path):
        return make_response("No staged data found.", 400)
    try:
        indexFile(staged_file_path)
    except Exception as e:
        print("Error: ", e, flush=True)
        return make_response("Indexing failed.", 500)
    return {"message": "Indexing finished."}, 201


@app.route("/search", methods=["GET"])
def lookup():
    query = request.args.get("query")
    try:
        page = int(request.args.get("page"))
    except:
        page = 0

    if query is None or query == "":
        return make_response("No query provided", 400)

    searcher, analyzer = init_searcher()
    if searcher is None or analyzer is None:
        return make_response("Search service not available.", 503)

    df, hits, sentiments = run(
        searcher, analyzer, query, page=page, calculate_sentiment=True
    )
    del searcher
    del analyzer
    print(sentiments, flush=True)
    return {
        "message": "Ok!",
        "data": df.to_dict(orient="records"),
        "hits": hits,
        "sentiments": sentiments,
    }, 200
