"""
Indexing using PyLucene, example code
"""
import os
import csv
import pandas as pd
from pathlib import Path

import lucene

from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import IndexOptions, IndexWriter, IndexWriterConfig
from org.apache.lucene.store import MMapDirectory, NIOFSDirectory


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

    sentimentField = FieldType()
    # textFieldType.setIndexed(True)
    sentimentField.setStored(True)
    sentimentField.setTokenized(True)
    sentimentField.setIndexOptions(IndexOptions.DOCS_AND_FREQS)

    return usernameField, tweetField, sentimentField


def indexFile(path: str):
    if not path.endswith(".csv") and not path.endswith(".json"):
        return
    writerConfig = IndexWriterConfig(StandardAnalyzer())
    writerConfig.setOpenMode(IndexWriterConfig.OpenMode.CREATE_OR_APPEND)
    fsDir = MMapDirectory(Paths.get("index"))
    writer = IndexWriter(fsDir, writerConfig)
    csv = path.endswith(".csv")

    usernameField, tweetField, sentimentField = initFields()

    if csv:
        df = pd.read_csv(path, header=0, encoding="latin-1")
        for index, row in df.iterrows():
            doc = Document()
            name = row["UserName"]
            tweet = row["OriginalTweet"]
            sentiment = row["Sentiment"]
            doc.add(Field("username", name, usernameField))
            doc.add(Field("tweet", tweet, tweetField))
            doc.add(Field("sentiment", sentiment, sentimentField))
            writer.addDocument(doc)
    else:
        df = pd.read_json(path)
        print(df, flush=True)
        for index, row in df.iterrows():
            doc = Document()
            name = row["user"]
            tweet = row["tweet"]
            sentiment = row["sentiment"]
            doc.add(Field("username", name, usernameField))
            doc.add(Field("tweet", tweet, tweetField))
            doc.add(Field("sentiment", sentiment, sentimentField))
            writer.addDocument(doc)

    writer.commit()
    writer.close()


def indexData(dataPath: str):
    for path, subDirs, filenames in os.walk(dataPath):
        for filename in filenames:
            fullPath = path + "/" + filename
            indexFile(fullPath)


if __name__ == "__main__":
    lucene.initVM(vmargs=["-Djava.awt.headless=true"])
    data_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, "data", "coronanlp")
    )
    indexData(data_path)
