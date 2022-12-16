"""
Indexing using PyLucene, example code
"""
import os, csv, sys
from pathlib import Path
import pandas as pd

import lucene

from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import (
    IndexOptions,
    IndexWriter,
    IndexWriterConfig,
    DirectoryReader,
)
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import MMapDirectory, NIOFSDirectory
from org.apache.lucene.search import IndexSearcher

PAGE_SIZE = 25


def run(
    searcher,
    analyzer,
    request="trending",
    field="tweet",
    page=0,
    calculate_sentiment=False,
):
    query = QueryParser(field, analyzer).parse(request)
    cols = ["score", "username", "tweet", "sentiment"]

    scoreDocs = searcher.search(query, 100_000).scoreDocs
    df = pd.DataFrame(columns=cols)

    query_sentiment = None
    if calculate_sentiment:
        query_sentiment = sentiment_from_query(searcher, scoreDocs)

    for i in range(page * PAGE_SIZE, (page + 1) * PAGE_SIZE):
        if i >= len(scoreDocs):
            break
        scoreDoc = scoreDocs[i]
        doc = searcher.doc(scoreDoc.doc)
        tweet = doc.get("tweet")
        sentiment = doc.get("sentiment")
        username = doc.get("username")
        score = scoreDoc.score
        df = pd.concat(
            [
                df,
                pd.DataFrame([[score, username, tweet, sentiment]], columns=cols),
            ],
            ignore_index=True,
        )
    return df, len(scoreDocs), query_sentiment


def sentiment_from_query(searcher, scoreDocs):
    sentiment_by_score = 0
    sentiment_no_score = 0
    max_score = scoreDocs[0].score
    for scoreDoc in scoreDocs:
        doc = searcher.doc(scoreDoc.doc)
        doc_sentiment = doc.get("sentiment").lower()
        sentiment_by_score += get_sentiment_by_score(
            doc_sentiment, scoreDoc.score, max_score
        )
        sentiment_no_score += get_sentiment_no_score(doc_sentiment)
    return sentiment_by_score / len(scoreDocs), sentiment_no_score / len(scoreDocs)


def get_sentiment_by_score(doc_sentiment, score, max_score):
    if doc_sentiment == "extremely negative":
        sentiment = -0.5
    elif doc_sentiment == "negative":
        sentiment = -0.25
    elif doc_sentiment == "positive":
        sentiment = 0.25
    elif doc_sentiment == "extremely positive":
        sentiment = 0.5
    else:
        sentiment = 0
    return (sentiment * (score / max_score)) + 0.5


def get_sentiment_no_score(doc_sentiment):
    if doc_sentiment == "extremely negative":
        sentiment = 0
    elif doc_sentiment == "negative":
        sentiment = 0.25
    elif doc_sentiment == "positive":
        sentiment = 0.75
    elif doc_sentiment == "extremely positive":
        sentiment = 1
    else:
        sentiment = 0.5
    return sentiment


if __name__ == "__main__":
    request = "trending"
    lucene.initVM(vmargs=["-Djava.awt.headless=true"])
    Paths.get("index")
    path = Paths.get(os.path.join(os.path.dirname(__file__), "index"))
    directory = NIOFSDirectory(path)
    searcher = IndexSearcher(DirectoryReader.open(directory))
    analyzer = StandardAnalyzer()
    run(searcher, analyzer, request)
    del searcher
