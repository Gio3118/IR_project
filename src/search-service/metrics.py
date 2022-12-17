import os
import requests
import pandas as pd
import shutil

import lucene
from java.nio.file import Paths

from org.apache.lucene.store import NIOFSDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import (
    DirectoryReader,
    IndexOptions,
    IndexWriter,
    IndexWriterConfig,
)
from org.apache.lucene.store import MMapDirectory

import matplotlib.pyplot as plt


scraper_serivce_uri = "http://scraping-service:5000"
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "data"))

lucene.initVM(vmargs=["-Djava.awt.headless=true"])
path = Paths.get(os.path.join(os.path.dirname(__file__), "tmpindex"))

searcher = None
analyzer = None


def metrics():
    shutil.rmtree("tmpindex", ignore_errors=True)
    world_cup = "World cup"
    elon_musk = "Elon Musk"
    bitcoin = "Bitcoin"
    # Index some search results
    # req = requests.get(
    #     scraper_serivce_uri + "/tweets", params={"query": world_cup, "limit": 250}
    # )
    indexFile(os.path.join(DATA, "scraper-staging", "staging.json"), "tmpindex")
    df_world_cup = pd.read_json(os.path.join(DATA, "scraper-staging", "staging.json"))
    # req = requests.get(
    #     scraper_serivce_uri + "/tweets", params={"query": elon_musk, "limit": 250}
    # )
    # indexFile(os.path.join(DATA, "scraper-staging", "staging.json"), "tmpindex")
    # df_elon_musk = pd.read_json(os.path.join(DATA, "scraper-staging", "staging.json"))
    # req = requests.get(
    #     scraper_serivce_uri + "/tweets", params={"query": bitcoin, "limit": 250}
    # )
    # indexFile(os.path.join(DATA, "scraper-staging", "staging.json"), "tmpindex")

    global searcher, analyzer

    directory = NIOFSDirectory(path)
    searcher = IndexSearcher(DirectoryReader.open(directory))
    analyzer = StandardAnalyzer()

    curve(world_cup, df_world_cup)


def curve(query, df):
    # Do the lucene search
    search_df = search(query)
    # Calculate precision and recall
    # Precision = TP / (TP + FP)
    # Recall = TP / (TP + FN)
    df_merged = pd.concat([search_df, df], axis=1)
    df_merged.columns = ["username_x", "tweet_x", "username_y", "tweet_y"]
    print(df_merged)
    df_merged["in_both"] = df_merged["tweet_x"].isin(df_merged["tweet_y"])
    print(df_merged)

    # Initialize lists to store the recall and precision values
    recall_vals = []
    precision_vals = []

    # Iterate over the range for the number of tweets retrieved
    for i in range(len(df_merged["in_both"])):
        # Top-i PR curve
        # Calculate the recall and precision for the current value

        true_positives = df_merged["in_both"][:i].sum()

        recall = true_positives / (true_positives + false_negatives)
        precision = true_positives / (true_positives + false_positives)
        # Append the recall and precision values to the lists
        recall_vals.append(recall)
        precision_vals.append(precision)

    # Plot the recall-precision curve
    print(recall_vals, precision_vals, flush=True)
    plt.plot(recall_vals, precision_vals)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Recall-Precision Curve")
    plt.savefig(os.path.join(DATA, "metrics", "pr_curve.png"))


def search(request):
    query = QueryParser("tweet", analyzer).parse(request)
    cols = ["username", "tweet"]

    scoreDocs = searcher.search(query, 100_000).scoreDocs
    df = pd.DataFrame(columns=cols)

    for scoreDoc in scoreDocs:
        doc = searcher.doc(scoreDoc.doc)
        tweet = doc.get("tweet")
        username = doc.get("username")
        df = pd.concat(
            [
                df,
                pd.DataFrame([[username, tweet]], columns=cols),
            ],
            ignore_index=True,
        )
    return df


def initFields():
    # field types
    usernameField = FieldType()
    # nameFieldType.setIndexed(False)
    usernameField.setStored(True)
    usernameField.setTokenized(True)
    usernameField.setIndexOptions(IndexOptions.DOCS_AND_FREQS)

    tweetField = FieldType()
    # textFieldType.setIndexed(True)
    tweetField.setStored(True)
    tweetField.setTokenized(True)
    tweetField.setIndexOptions(IndexOptions.DOCS_AND_FREQS)

    return usernameField, tweetField


def indexFile(path: str, index_path: str = "index"):
    if not path.endswith(".csv") and not path.endswith(".json"):
        return
    writerConfig = IndexWriterConfig(StandardAnalyzer())
    writerConfig.setOpenMode(IndexWriterConfig.OpenMode.CREATE_OR_APPEND)
    fsDir = MMapDirectory(Paths.get(index_path))
    writer = IndexWriter(fsDir, writerConfig)

    usernameField, tweetField = initFields()

    df = pd.read_json(path)
    for index, row in df.iterrows():
        doc = Document()
        name = row["user"]
        tweet = row["tweet"]
        doc.add(Field("username", name, usernameField))
        doc.add(Field("tweet", tweet, tweetField))
        writer.addDocument(doc)

    writer.commit()
    writer.close()


if __name__ == "__main__":
    metrics()
