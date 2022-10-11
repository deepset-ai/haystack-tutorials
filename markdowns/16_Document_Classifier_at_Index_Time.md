---
layout: tutorial
colab: https://colab.research.google.com/github/deepset-ai/haystack-tutorials/blob/main/tutorials/16_Document_Classifier_at_Index_Time.ipynb
toc: True
title: "Document Classifier at Index Time"
last_updated: 2022-10-11
level: "intermediate"
weight: 85
description: Generate and attach classification labels to your Documents when indexing.
category: "QA"
aliases: ['/tutorials/doc-class-index']
---
    

# Extending your Metadata using DocumentClassifiers at Index Time

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepset-ai/haystack-tutorials/blob/main/tutorials/16_Document_Classifier_at_Index_Time.ipynb)

With DocumentClassifier it's possible to automatically enrich your documents with categories, sentiments, topics or whatever metadata you like. This metadata could be used for efficient filtering or further processing. Say you have some categories your users typically filter on. If the documents are tagged manually with these categories, you could automate this process by training a model. Or you can leverage the full power and flexibility of zero shot classification. All you need to do is pass your categories to the classifier, no labels required. This tutorial shows how to integrate it in your indexing pipeline.

DocumentClassifier adds the classification result (label and score) to Document's meta property.
Hence, we can use it to classify documents at index time. \
The result can be accessed at query time: for example by applying a filter for "classification.label".

This tutorial will show you how to integrate a classification model into your preprocessing steps and how you can filter for this additional metadata at query time. In the last section we show how to put it all together and create an indexing pipeline.


```bash
%%bash

# Install the latest main of Haystack
pip install --upgrade pip
pip install git+https://github.com/deepset-ai/haystack.git#egg=farm-haystack[colab,ocr]

!wget --no-check-certificate https://dl.xpdfreader.com/xpdf-tools-linux-4.04.tar.gz
!tar -xvf xpdf-tools-linux-4.04.tar.gz && sudo cp xpdf-tools-linux-4.04/bin64/pdftotext /usr/local/bin

apt install libgraphviz-dev
pip install pygraphviz
```

## Logging

We configure how logging messages should be displayed and which log level should be used before importing Haystack.
Example log message:
INFO - haystack.utils.preprocessing -  Converting data/tutorial1/218_Olenna_Tyrell.txt
Default log level in basicConfig is WARNING so the explicit parameter is not necessary but can be changed easily:


```python
import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)
```

## Read and preprocess documents



```python
from haystack.utils import fetch_archive_from_http


# This fetches some sample files to work with
doc_dir = "data/tutorial16"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/preprocessing_tutorial16.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)
```


```python
from haystack.nodes import PreProcessor
from haystack.utils import convert_files_to_docs

# note that you can also use the document classifier before applying the PreProcessor, e.g. before splitting your documents
all_docs = convert_files_to_docs(dir_path=doc_dir)
preprocessor_sliding_window = PreProcessor(split_overlap=3, split_length=10, split_respect_sentence_boundary=False)
docs_sliding_window = preprocessor_sliding_window.process(all_docs)
```

## Apply DocumentClassifier

We can enrich the document metadata at index time using any transformers document classifier model. While traditional classification models are trained to predict one of a few "hard-coded" classes and required a dedicated training dataset, zero-shot classification is super flexible and you can easily switch the classes the model should predict on the fly. Just supply them via the labels param.
Here we use a zero shot model that is supposed to classify our documents in 'music', 'natural language processing' and 'history'. Feel free to change them for whatever you like to classify. \
These classes can later on be accessed at query time.


```python
from haystack.nodes import TransformersDocumentClassifier


doc_classifier = TransformersDocumentClassifier(
    model_name_or_path="cross-encoder/nli-distilroberta-base",
    task="zero-shot-classification",
    labels=["music", "natural language processing", "history"],
    batch_size=16,
)
```


```python
# we can also use any other transformers model besides zero shot classification

# doc_classifier_model = 'bhadresh-savani/distilbert-base-uncased-emotion'
# doc_classifier = TransformersDocumentClassifier(model_name_or_path=doc_classifier_model, batch_size=16, use_gpu=-1)
```


```python
# we could also specifiy a different field we want to run the classification on

# doc_classifier = TransformersDocumentClassifier(model_name_or_path="cross-encoder/nli-distilroberta-base",
#    task="zero-shot-classification",
#    labels=["music", "natural language processing", "history"],
#    batch_size=16, use_gpu=-1,
#    classification_field="description")
```


```python
# classify using gpu, batch_size makes sure we do not run out of memory
classified_docs = doc_classifier.predict(docs_sliding_window)
```


```python
# let's see how it looks: there should be a classification result in the meta entry containing labels and scores.
print(classified_docs[0].to_dict())
```

## Indexing

### Start an Elasticsearch server
You can start Elasticsearch on your local machine instance using Docker. If Docker is not readily available in your environment (eg., in Colab notebooks), then you can manually download and execute Elasticsearch from source.


```python
# Recommended: Start Elasticsearch using Docker via the Haystack utility function
from haystack.utils import launch_es

launch_es()
```

### Start an Elasticsearch server in Colab

If Docker is not readily available in your environment (e.g. in Colab notebooks), then you can manually download and execute Elasticsearch from source.


```bash
%%bash

wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.9.2-linux-x86_64.tar.gz -q
tar -xzf elasticsearch-7.9.2-linux-x86_64.tar.gz
chown -R daemon:daemon elasticsearch-7.9.2
sudo -u daemon -- elasticsearch-7.9.2/bin/elasticsearch -d
```


```bash
%%bash --bg

sudo -u daemon -- elasticsearch-7.9.2/bin/elasticsearch
```


```python
# Connect to Elasticsearch
import os
import time

from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore

# Wait 30 seconds only to be sure Elasticsearch is ready before continuing
time.sleep(30)

# Get the host where Elasticsearch is running, default to localhost
host = os.environ.get("ELASTICSEARCH_HOST", "localhost")

document_store = ElasticsearchDocumentStore(host=host, username="", password="", index="document")
```


```python
# Now, let's write the docs to our DB.
document_store.delete_all_documents()
document_store.write_documents(classified_docs)
```


```python
# check if indexed docs contain classification results
test_doc = document_store.get_all_documents()[0]
print(
    f'document {test_doc.id} with content \n\n{test_doc.content}\n\nhas label {test_doc.meta["classification"]["label"]}'
)
```

## Querying the data

All we have to do to filter for one of our classes is to set a filter on "classification.label".


```python
# Initialize QA-Pipeline
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import FARMReader, BM25Retriever


retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
pipe = ExtractiveQAPipeline(reader, retriever)
```


```python
## Voilà! Ask a question while filtering for "music"-only documents
prediction = pipe.run(
    query="What is heavy metal?",
    params={"Retriever": {"top_k": 10, "filters": {"classification.label": ["music"]}}, "Reader": {"top_k": 5}},
)
```


```python
from haystack.utils import print_answers


print_answers(prediction, details="high")
```

## Wrapping it up in an indexing pipeline


```python
from pathlib import Path
from haystack.pipelines import Pipeline
from haystack.nodes import TextConverter, PreProcessor, FileTypeClassifier, PDFToTextConverter, DocxToTextConverter


file_type_classifier = FileTypeClassifier()
text_converter = TextConverter()
pdf_converter = PDFToTextConverter()
docx_converter = DocxToTextConverter()

indexing_pipeline_with_classification = Pipeline()
indexing_pipeline_with_classification.add_node(
    component=file_type_classifier, name="FileTypeClassifier", inputs=["File"]
)
indexing_pipeline_with_classification.add_node(
    component=text_converter, name="TextConverter", inputs=["FileTypeClassifier.output_1"]
)
indexing_pipeline_with_classification.add_node(
    component=pdf_converter, name="PdfConverter", inputs=["FileTypeClassifier.output_2"]
)
indexing_pipeline_with_classification.add_node(
    component=docx_converter, name="DocxConverter", inputs=["FileTypeClassifier.output_4"]
)
indexing_pipeline_with_classification.add_node(
    component=preprocessor_sliding_window,
    name="Preprocessor",
    inputs=["TextConverter", "PdfConverter", "DocxConverter"],
)
indexing_pipeline_with_classification.add_node(
    component=doc_classifier, name="DocumentClassifier", inputs=["Preprocessor"]
)
indexing_pipeline_with_classification.add_node(
    component=document_store, name="DocumentStore", inputs=["DocumentClassifier"]
)
# Uncomment the following to generate the pipeline image
# indexing_pipeline_with_classification.draw("index_time_document_classifier.png")

document_store.delete_documents()
txt_files = [f for f in Path(doc_dir).iterdir() if f.suffix == ".txt"]
pdf_files = [f for f in Path(doc_dir).iterdir() if f.suffix == ".pdf"]
docx_files = [f for f in Path(doc_dir).iterdir() if f.suffix == ".docx"]
indexing_pipeline_with_classification.run(file_paths=txt_files)
indexing_pipeline_with_classification.run(file_paths=pdf_files)
indexing_pipeline_with_classification.run(file_paths=docx_files)

document_store.get_all_documents()[0]
```


```python
# we can store this pipeline and use it from the REST-API
indexing_pipeline_with_classification.save_to_yaml("indexing_pipeline_with_classification.yaml")
```

## About us

This [Haystack](https://github.com/deepset-ai/haystack/) notebook was made with love by [deepset](https://deepset.ai/) in Berlin, Germany

We bring NLP to the industry via open source!  
Our focus: Industry specific language models & large scale QA systems.  
  
Some of our other work: 
- [German BERT](https://deepset.ai/german-bert)
- [GermanQuAD and GermanDPR](https://deepset.ai/germanquad)
- [FARM](https://github.com/deepset-ai/FARM)

Get in touch:
[Twitter](https://twitter.com/deepset_ai) | [LinkedIn](https://www.linkedin.com/company/deepset-ai/) | [Discord](https://haystack.deepset.ai/community/join) | [GitHub Discussions](https://github.com/deepset-ai/haystack/discussions) | [Website](https://deepset.ai)

By the way: [we're hiring!](https://www.deepset.ai/jobs)

