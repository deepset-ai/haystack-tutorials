---
layout: tutorial
featured: False
colab: https://colab.research.google.com/github/deepset-ai/haystack-tutorials/blob/main/tutorials/03_Scalable_QA_System.ipynb
toc: True
title: "Build a Scalable Question Answering System"
last_updated: 2023-01-24
level: "beginner"
weight: 15
description: Create a scalable Retriever Reader pipeline that uses an ElasticsearchDocumentStore.
category: "QA"
aliases: []
download: "/downloads/03_Scalable_QA_System.ipynb"
completion_time: 20 min
created_at: 2023-01-11
---
    


- **Level**: Beginner
- **Time to complete**: 20 minutes
- **Nodes Used**: `ElasticsearchDocumentStore`, `BM25Retriever`, `FARMReader`
- **Goal**: After completing this tutorial, you'll have built a scalable search system that runs on text files and can answer questions about Game of Thrones. You'll then be able to expand this system for your needs.


## Overview

Learn how to set up a question answering system that can search through complex knowledge bases and highlight answers to questions such as "Who is the father of Arya Stark?". In this tutorial, we'll work on a set of Wikipedia pages about Game of Thrones, but you can adapt it to search through internal wikis or a collection of financial reports, for example.

This tutorial introduces you to all the concepts needed to build such a question answering system. It also uses Haystack components, such as indexing pipelines, querying pipelines, and DocumentStores backed by external database services.

Let's learn how to build a question answering system and discover more about the marvelous seven kingdoms!


## Preparing the Colab Environment

- [Enable GPU Runtime](https://docs.haystack.deepset.ai/docs/enabling-gpu-acceleration#enabling-the-gpu-in-colab)


## Installing Haystack

To start, let's install the latest release of Haystack with `pip`:


```bash
%%bash

pip install --upgrade pip
pip install farm-haystack[colab]
```

Set the logging level to INFO:


```python
import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)
```

## Initializing the ElasticsearchDocumentStore

A DocumentStore stores the Documents that the question answering system uses to find answers to your questions. Here, we're using the [`ElasticsearchDocumentStore`](https://docs.haystack.deepset.ai/reference/document-store-api#module-elasticsearch) which connects to a running Elasticsearch service. It's a fast and scalable text-focused storage option. This service runs independently from Haystack and persists even after the Haystack program has finished running. To learn more about the DocumentStore and the different types of external databases that we support, see [DocumentStore](https://docs.haystack.deepset.ai/docs/document_store).

1. Download, extract, and set the permissions for the Elasticsearch installation image:


```bash
%%bash

wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.9.2-linux-x86_64.tar.gz -q
tar -xzf elasticsearch-7.9.2-linux-x86_64.tar.gz
chown -R daemon:daemon elasticsearch-7.9.2
```

2. Start the server:


```bash
%%bash --bg

sudo -u daemon -- elasticsearch-7.9.2/bin/elasticsearch
```

If you are working in an environment where Docker is available, you can also start Elasticsearch using Docker. You can do this manually, or using our [`launch_es()`](https://docs.haystack.deepset.ai/reference/utils-api#module-doc_store) utility function.

3. Wait 30 seconds for the server to fully start up:


```python
import time
time.sleep(30)
```

4. Initialize the ElasticsearchDocumentStore:



```python
from haystack.utils import launch_es
launch_es()
```


```python
import os
from haystack.document_stores import ElasticsearchDocumentStore

# Get the host where Elasticsearch is running, default to localhost
host = os.environ.get("ELASTICSEARCH_HOST", "localhost")

document_store = ElasticsearchDocumentStore(
    host=host,
    username="",
    password="",
    index="document"
)
```

ElasticsearchDocumentStore is up and running and ready to store the Documents.

## Indexing Documents with a Pipeline

The next step is adding the files to the DocumentStore. The indexing pipeline turns your files into Document objects and writes them to the DocumentStore. Our indexing pipeline will have two nodes: `TextConverter`, which turns `.txt` files into Haystack `Document` objects, and `PreProcessor`, which cleans and splits the text within a `Document`.

Once we combine these nodes into a pipeline, the pipeline will ingest `.txt` file paths, preprocess them, and write them into the DocumentStore.


1. Download 517 articles from the Game of Thrones Wikipedia. You can find them in *data/build_a_scalable_question_answering_system* as a set of *.txt* files.


```python
from haystack.utils import fetch_archive_from_http

doc_dir = "data/build_a_scalable_question_answering_system"

fetch_archive_from_http(
    url="https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt3.zip", 
    output_dir=doc_dir
)
```

2. Initialize the pipeline, TextConverter, and PreProcessor:


```python
from haystack import Pipeline
from haystack.nodes import TextConverter, PreProcessor

indexing_pipeline = Pipeline()
text_converter = TextConverter()
preprocessor = PreProcessor(
    clean_whitespace=True,
    clean_header_footer=True,
    clean_empty_lines=True,
    split_by="word",
    split_length=200,
    split_overlap=20,
    split_respect_sentence_boundary=True,
)

```

To learn more about the parameters of the `PreProcessor`, see [Usage](https://docs.haystack.deepset.ai/docs/preprocessor#usage). To understand why document splitting is important for your question answering system's performance, see [Document Length](https://docs.haystack.deepset.ai/docs/optimization#document-length).

2. Add the nodes into an indexing pipeline. You should provide the `name` or `name`s of preceding nodes as the `input` argument. Note that in an indexing pipeline, the input to the first node is `File`.


```python
import os

indexing_pipeline.add_node(component=text_converter, name="TextConverter", inputs=["File"])
indexing_pipeline.add_node(component=preprocessor, name="PreProcessor", inputs=["TextConverter"])
indexing_pipeline.add_node(component=document_store, name="DocumentStore", inputs=["PreProcessor"])

```

3. Run the indexing pipeline to write the text data into the DocumentStore:


```python
files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
indexing_pipeline.run_batch(file_paths=files_to_index)
```

The code in this tutorial uses Game of Thrones data, but you can also supply your own `.txt` files and index them in the same way.

As an alternative, you can cast you text data into [Document objects](https://docs.haystack.deepset.ai/docs/documents_answers_labels#document) and write them into the DocumentStore using [`DocumentStore.write_documents()`](https://docs.haystack.deepset.ai/reference/document-store-api#basedocumentstorewrite_documents).

Now that the Documents are in the DocumentStore, let's initialize the nodes we want to use in our query pipeline.

## Initializing the Retriever

Our query pipeline is going to use a Retriever, so we need to initialize it. A Retriever sifts through all the Documents and returns only those that are relevant to the question. This tutorial uses the BM25Retriever. This is the recommended Retriever for a question answering system like the one we're creating. For more Retriever options, see [Retriever](https://docs.haystack.deepset.ai/docs/retriever).


```python
from haystack.nodes import BM25Retriever

retriever = BM25Retriever(document_store=document_store)
```

The BM25Retriever is initialized and ready for the pipeline.

## Initializing the Reader

Our query pipeline also needs a Reader, so we'll initialize it next. A Reader scans the texts it received from the Retriever and extracts the top answer candidates. Readers are based on powerful deep learning models but are much slower than Retrievers at processing the same amount of text. This tutorials uses a FARMReader with a base-sized RoBERTa question answering model called [`deepset/roberta-base-squad2`](https://huggingface.co/deepset/roberta-base-squad2). It's a good all-round model to start with. To find a model that's best for your use case, see [Models](https://docs.haystack.deepset.ai/docs/reader#models).


```python
from haystack.nodes import FARMReader

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
```

## Creating the Retriever-Reader Pipeline

You can combine the Reader and Retriever in a querying pipeline using the `Pipeline` class. The combination of the two speeds up processing because the Reader only processes the Documents that it received from the Retriever. 

Initialize the `Pipeline` object and add the Retriever and Reader as nodes. You should provide the `name` or `name`s of preceding nodes as the input argument. Note that in a querying pipeline, the input to the first node is `Query`.


```python
from haystack import Pipeline

querying_pipeline = Pipeline()
querying_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
querying_pipeline.add_node(component=reader, name="Reader", inputs=["Retriever"])

```

That's it! Your pipeline's ready to answer your questions!

## Asking a Question

1. Use the pipeline's `run()` method to ask a question. The query argument is where you type your question. Additionally, you can set the number of documents you want the Reader and Retriever to return using the `top-k` parameter. To learn more about setting arguments, see [Arguments](https://docs.haystack.deepset.ai/docs/pipelines#arguments). To understand the importance of the `top-k` parameter, see [Choosing the Right top-k Values](https://docs.haystack.deepset.ai/docs/optimization#choosing-the-right-top-k-values).



```python
prediction = querying_pipeline.run(
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

2. Print out the answers the pipeline returns:


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

And there you have it! Congratulations on building a scalable machine learning based question answering system!

# Next Steps

To learn how to improve the performance of the Reader, see [Fine-Tune a Reader](https://haystack.deepset.ai/tutorials/02_finetune_a_model_on_your_data).

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

