---
layout: tutorial
colab: https://colab.research.google.com/github/deepset-ai/haystack-tutorials/blob/main/tutorials/01_Basic_QA_Pipeline.ipynb
toc: True
title: "Build Your First Question Answering System"
last_updated: 2022-12-30
level: "beginner"
weight: 10
description: Get Started by creating a Retriever Reader pipeline.
category: "QA"
aliases: ['/tutorials/first-qa-system', '/tutorials/without-elasticsearch', '/tutorials/03_basic_qa_pipeline_without_elasticsearch']
download: "/downloads/01_Basic_QA_Pipeline.ipynb"
---
    


> We've modified this first tutorial to make it simpler to start with. If you're looking for a Question Answering tutorial that uses a DocumentStore such as Elasticsearch, go to our new [Build a Scalable Question Answering System](https://haystack.deepset.ai/tutorials/03_Scalable_QA_System) tutorial

- **Level**: Beginner
- **Time to complete**: 15 minutes
- **Nodes Used**: `InMemoryDocumentStore`, `BM25Retriever`, `FARMReader`
- **Goal**: After completing this tutorial, you will have learned about the Reader and Retriever, and built a question answering pipeline that can answer questions about the Game of Thrones series.


## Overview

Let's learn how to build a question answering system using Haystack's DocumentStore, Retriever, and Reader. Given a question like "Who is the father of Arya Stark?", this program will search through a knowledge base and look for a fitting answer.

While the documents we are using in this tutorial are all to do with Game of Thrones, the question answering system can work in many domains if you provide the documents. For example, you could add your company's internal wikis, or a collection of financial reports and still receive answers to questions on these topics.

To help you get started quicker, we have simplified certain steps in this tutorial. For example, Document preparation and pipeline initialization are handled by ready-made classes that replace lines of initialization code. But don't worry! This doesn't affect how well the question answering system performs.


## Preparing the Colab Environment

- [Enable GPU Runtime in Colab](https://docs.haystack.deepset.ai/docs/enabling-gpu-acceleration#enabling-the-gpu-in-colab)
- [Set logging level to INFO](https://docs.haystack.deepset.ai/docs/set-the-logging-level)


## Installing Haystack

To start, let's install the latest release of Haystack with `pip`:


```bash
%%bash

pip install --upgrade pip
pip install farm-haystack[colab]
```

## Initializing the DocumentStore

A DocumentStore stores the Documents that the question answering system uses to find answers to your questions. Here we are using the `InMemoryDocumentStore` which is the simplest DocumentStore to get started with. It requires no external dependencies and is a good option for smaller projects and debugging. However, it does not scale up so well to larger Document collections. To learn more about the DocumentStore and the different types of external databases that we support, see [DocumentStore](https://docs.haystack.deepset.ai/docs/document_store).


```python
from haystack.document_stores import InMemoryDocumentStore

document_store = InMemoryDocumentStore(use_bm25=True)
```

## Preparing Documents

1. Download 517 articles from the Game of Thrones Wikipedia. You can find them in `data/tutorial1` as a set of `.txt` files.


```python
from haystack.utils import fetch_archive_from_http

doc_dir = "data/tutorial1"

fetch_archive_from_http(
    url="https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt1.zip",
    output_dir=doc_dir
)
```

2. Use the `TextIndexingPipeline` to convert the files you just downloaded into Haystack [Document objects](https://docs.haystack.deepset.ai/docs/documents_answers_labels#document) and write them into the DocumentStore.


```python
import os
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
# from text_indexing_pipeline import TextIndexingPipeline

files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
indexing_pipeline = TextIndexingPipeline(document_store)
indexing_pipeline.run_batch(file_paths=files_to_index)


```

While the default code in this tutorial uses Game of Thrones data, you can also supply your own `.txt` files and index them in the same way.

As an alternative, you can cast you text data into [Document objects](https://docs.haystack.deepset.ai/docs/documents_answers_labels#document) and write them into the DocumentStore using `DocumentStore.write_documents()`.

## Initializing the Retriever

Retrievers sift through all the Documents and return only those that it thinks might be relevant to the question. Here we are using the BM25 algorithm. For more Retriever options, see [Retriever](https://docs.haystack.deepset.ai/docs/retriever).


```python
from haystack.nodes import BM25Retriever

retriever = BM25Retriever(document_store=document_store)
```

## Initializing the Reader

A Reader scans the texts returned by Retrievers in detail and extracts the top answer candidates. Readers are based on powerful deep learning models but are much slower than Retrievers at processing the same amount of text. Here we are using a base sized RoBERTa question answering model called [`deepset/roberta-base-squad2`](https://huggingface.co/deepset/roberta-base-squad2). To find out what model works best for your use case, see [Models](https://haystack.deepset.ai/pipeline_nodes/reader#models).


```python
from haystack.nodes import FARMReader

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
```

## Creating the Retriever-Reader Pipeline

The `ExtractiveQAPipeline` connects the Reader and Retriever. The combination of the two speeds up processing because the Reader only processes the Documents that the Retriever has passed on.


```python
from haystack.pipelines import ExtractiveQAPipeline

pipe = ExtractiveQAPipeline(reader, retriever)
```

## Asking a Question

1. Use the pipeline `run()` method to ask a question. The query argument is where you type your question. Additionally, you can set the number of documents you want the Reader and Retriever to return using the `top-k` parameter. To learn more about setting arguments, see [Arguments](https://docs.haystack.deepset.ai/docs/pipelines#arguments). To understand the importance of the `top-k` parameter, see [Choosing the Right top-k Values](https://docs.haystack.deepset.ai/docs/optimization#choosing-the-right-top-k-values).



```python
prediction = pipe.run(
    query="Who is the father of Arya Stark?",
    params={
        "Retriever": {"top_k": 10},
        "Reader": {"top_k": 5}
    }
)
```

Here are some questions you could try out:
- Who is the father of Arya Stark?
- Who created the Dothraki vocabulary?
- Who is the sister of Sansa?

2. The answers returned by the pipeline can be printed out directly:


```python
from pprint import pprint

pprint(prediction)
```

3. Simplify the printed answers:


```python
from haystack.utils import print_answers

print_answers(
    prediction,
    details="minimum" ## Choose from `minimum`, `medium` and `all`
)
```

And there you have it! Congratulations on building your first machine learning based question answering system!

# Next Steps

Check out [Build a Scalable Question Answering System](https://haystack.deepset.ai/tutorials/03_Scalable_QA_System) to learn how to make a more advanced question answering system that uses an Elasticsearch backed DocumentStore and makes more use of the flexibility that pipelines offer.

## About us

This [Haystack](https://github.com/deepset-ai/haystack/) notebook was made with love by [deepset](https://deepset.ai/) in Berlin, Germany

We bring NLP to the industry via open source!  
Our focus: Industry specific language models & large scale QA systems.  
  
Some of our other work: 
- [German BERT](https://deepset.ai/german-bert)
- [GermanQuAD and GermanDPR](https://deepset.ai/germanquad)

Get in touch:
[Twitter](https://twitter.com/deepset_ai) | [LinkedIn](https://www.linkedin.com/company/deepset-ai/) | [Discord](https://haystack.deepset.ai/community) | [GitHub Discussions](https://github.com/deepset-ai/haystack/discussions) | [Website](https://deepset.ai)

By the way: [we're hiring!](https://www.deepset.ai/jobs)

