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


def run(searcher, analyzer, request="trending", field="tweet", page=0):
    query = QueryParser(field, analyzer).parse(request)

    scoreDocs = searcher.search(query, 100_000).scoreDocs
    df = pd.DataFrame(columns=["score", "username", "tweet", "sentiment"])

    for i in range(page * PAGE_SIZE, (page + 1) * PAGE_SIZE):
        if i >= len(scoreDocs):
            break
        scoreDoc = scoreDocs[i]
        doc = searcher.doc(scoreDoc.doc)
        tweet = doc.get("tweet")
        sentiment = doc.get("sentiment")
        username = doc.get("username")
        score = scoreDoc.score
        df = df.append(
            {
                "score": score,
                "username": username,
                "tweet": tweet,
                "sentiment": sentiment,
            },
            ignore_index=True,
        )
    return df, len(scoreDocs)


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
