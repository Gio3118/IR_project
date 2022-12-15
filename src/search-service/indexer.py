'''
Indexing using PyLucene, example code
'''
import os
import csv
from pathlib import Path

import lucene

from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import (IndexOptions, IndexWriter, IndexWriterConfig)
from org.apache.lucene.store import MMapDirectory, NIOFSDirectory


# fsDir = MMapDirectory(Paths.get('index'))

#
# # Define field type
# t1 = FieldType()
# t1.setStored(True)
# t1.setIndexOptions(IndexOptions.DOCS)
#
# t2 = FieldType()
# t2.setStored(False)
# t2.setIndexOptions(IndexOptions.DOCS_AND_FREQS)
# print(f"{writer.numRamDocs()} docs found in index")
# # Add a document
# doc = Document()
# doc.add(Field('id', '418129481', t1))
#
# writer.addDocument(doc)
# print(f"{writer.numRamDocs()} docs found in index")

dataPath = "./data"

def indexData():
    writerConfig = IndexWriterConfig(StandardAnalyzer())
    writerConfig.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
    fsDir = MMapDirectory(Paths.get('index'))
    writer = IndexWriter(fsDir, writerConfig)

    #field types
    nameFieldType = FieldType()
    #nameFieldType.setIndexed(False)
    nameFieldType.setStored(True)
    nameFieldType.setTokenized(True)
    nameFieldType.setIndexOptions(IndexOptions.DOCS_AND_FREQS)


    textFieldType = FieldType()
    #textFieldType.setIndexed(True)
    textFieldType.setStored(True)
    textFieldType.setTokenized(True)
    textFieldType.setIndexOptions(IndexOptions.DOCS_AND_FREQS)

    sentimentFieldType = FieldType()
    #textFieldType.setIndexed(True)
    sentimentFieldType.setStored(True)
    sentimentFieldType.setTokenized(True)
    sentimentFieldType.setIndexOptions(IndexOptions.DOCS_AND_FREQS)

    for path, subDirs, filenames in os.walk(dataPath):
        print("path:")
        print(path)
        print("SUBDIRS")

        print(subDirs)
        print("FILENAMES:")
        print(filenames)

        for filename in filenames:
            fullPath = path + "/" + filename
            with open(fullPath) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                firstRow = True
                for row in csv_reader:
                    doc = Document()
                    if firstRow:
                        firstRow = False
                        continue
                    name = row[0]
                    text = row[4]
                    sentiment = row[5]
                    doc.add(Field('name', name , nameFieldType))
                    doc.add(Field('text', text , textFieldType))
                    doc.add(Field('sentiment', sentiment , sentimentFieldType))
                    writer.addDocument(doc)
    writer.commit()
    writer.close()

if __name__ == '__main__':
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    indexData()
