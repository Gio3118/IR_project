from flask import Flask, request, make_response, render_template
import requests
import os

DIST_DIR = os.path.join(os.path.dirname(__file__), "dist")
STATIC_DIR = os.path.join(DIST_DIR, "assets")
app = Flask(__name__, template_folder=DIST_DIR, static_folder=STATIC_DIR)


scraper_uri = "http://scraping-service:5000"


@app.route("/")
def index():
    return render_template("index.html")
