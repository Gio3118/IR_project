'''
Indexing using PyLucene, example code
'''
import os
from pathlib import Path

import lucene

from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import (IndexOptions, IndexWriter,
                                     IndexWriterConfig)
from org.apache.lucene.store import MMapDirectory

env = lucene.initVM(vmargs=['-Djava.awt.headless=true'])
fsDir = MMapDirectory(Paths.get('index'))
writerConfig = IndexWriterConfig(StandardAnalyzer())
writer = IndexWriter(fsDir, writerConfig)

# Define field type
t1 = FieldType()
t1.setStored(True)
t1.setIndexOptions(IndexOptions.DOCS)

t2 = FieldType()
t2.setStored(False)
t2.setIndexOptions(IndexOptions.DOCS_AND_FREQS)
print(f"{writer.numRamDocs()} docs found in index")
# Add a document
doc = Document()
doc.add(Field('id', '418129481', t1))
doc.add(Field('title', 'Lucene in Action', t1))
doc.add(Field('text', 'Hello Lucene, This is the first document', t2))
writer.addDocument(doc)
print(f"{writer.numRamDocs()} docs found in index")