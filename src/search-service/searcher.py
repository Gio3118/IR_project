'''
Indexing using PyLucene, example code
'''
import os, csv, sys
from pathlib import Path

import lucene

from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import (IndexOptions, IndexWriter, IndexWriterConfig, DirectoryReader)
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import MMapDirectory, NIOFSDirectory
from org.apache.lucene.search import IndexSearcher

dataPath = "./index"
command = "trending"
def run(searcher, analyzer):
    query = QueryParser("text", analyzer).parse(command)

    scoreDocs = searcher.search(query, 50).scoreDocs
    for scoreDoc in scoreDocs:
        print("scoreDoc:")
        print(scoreDoc)
        doc = searcher.doc(scoreDoc.doc)
        print ('sent:', doc.get("text"))

if __name__ == '__main__':
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    Paths.get('index')
    print(os.path.abspath(sys.argv[0]))
    path = Paths.get("./index")
    directory = NIOFSDirectory(path)
    searcher = IndexSearcher(DirectoryReader.open(directory))
    analyzer = StandardAnalyzer()
    run(searcher, analyzer)
    del searcher