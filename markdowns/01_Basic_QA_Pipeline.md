<!---
title: "Tutorial 1"
metaTitle: "Build Your First QA System"
metaDescription: ""
slug: "/docs/tutorial1"
date: "2020-09-03"
id: "tutorial1md"
--->

# Build Your First Question Answering System

- **Level**: Beginner
- **Time to complete**: 20 minutes
- **Prerequisites**: Prepare the Colab environment. See links below.
- **Nodes Used**: `ElasticsearchDocumentStore`, `BM25Retriever`
- **Goal**: After completing this tutorial, you will have built a question answering pipeline that can answer questions about the Game of Thrones series.

This tutorial teaches you how to set up a question answering system that can search through complex knowledge bases, such as an internal wiki or a collection of financial reports. We will work on a set of Wikipedia pages about Game of Thrones. Let's learn how to build a question answering system and discover more about the marvellous seven kingdoms!



## Preparing the Colab Environment

- [Enable GPU Runtime in GPU](https://docs.haystack.deepset.ai/v5.2-unstable/docs/enable-gpu-runtime-in-colab)
- [Check if GPU is Enabled](https://docs.haystack.deepset.ai/v5.2-unstable/docs/check-if-gpu-is-enabled)
- [Set logging level to INFO](https://docs.haystack.deepset.ai/v5.2-unstable/docs/set-the-logging-level)


## Installing Haystack

To start, let's install the latest release of Haystack with `pip`:


```bash
%%bash

pip install --upgrade pip
pip install git+https://github.com/deepset-ai/haystack.git#egg=farm-haystack[colab]
```

## Initializing the DocumentStore

A DocumentStore stores the documents that the question answering system uses to find answers to your questions. To learn more, see [DocumentStore](https://docs.haystack.deepset.ai/docs/document_store).

1. Download, extract, and set the permission for the Elasticsearch image:


```bash
%%bash

wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.9.2-linux-x86_64.tar.gz -q
tar -xzf elasticsearch-7.9.2-linux-x86_64.tar.gz
chown -R daemon:daemon elasticsearch-7.9.2
```

2. Start the Elasticsearch Server:


```bash
%%bash --bg

sudo -u daemon -- elasticsearch-7.9.2/bin/elasticsearch
```


```python
import time
time.sleep(30)
```

If you are working in an environment where Docker is available, you can also start Elasticsearch using Docker. You can do this [manually](https://docs.haystack.deepset.ai/docs/document_store#initialisation), or using our [`launch_es()`](https://docs.haystack.deepset.ai/reference/utils-api) utility function.

3. Initialize the `ElasticsearchDocumentStore` object in Haystack. Note that this will only successfully run if the Elasticsearch Server is fully started up and ready.


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

2. Convert the files you just downloaded into Haystack [Document objects](https://docs.haystack.deepset.ai/docs/documents_answers_labels#document) to write them into the DocumentStore. Apply the `clean_wiki_text` cleaning function to the text.


```python
from haystack.utils import clean_wiki_text, convert_files_to_docs
docs = convert_files_to_docs(
    dir_path=doc_dir,
    clean_func=clean_wiki_text,
    split_paragraphs=True
)
```

3. Write these Documents into the DocumentStore.


```python
# Now, let's write the dicts containing documents to our DB.
document_store.write_documents(docs)
```

While the default code in this tutorial uses Game of Thrones data, you can also supply your own. So long as your data adheres to the [input format](https://docs.haystack.deepset.ai/docs/document_store#input-format) or is cast into a [Document object](https://docs.haystack.deepset.ai/docs/documents_answers_labels#document), it can be written into the DocumentStore.

## Initializing the Retriever

Initialize the `BM25Retriever`. For more Retriever options, see [Retriever](https://docs.haystack.deepset.ai/docs/retriever)


```python
from haystack.nodes import BM25Retriever

retriever = BM25Retriever(document_store=document_store)
```

## Initializing the Reader

Initialize the `FARMReader` with the `deepset/robert-base-squad2` model. For more Reader options, see [Reader](https://docs.haystack.deepset.ai/docs/reader).


```python
from haystack.nodes import FARMReader

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
```

## Creating the Retriever-Reader Pipeline

The `ExtractiveQAPipeline` connects the Reader and Retriever. This makes the system fast because the Reader only processes the Documents that the Retriever has passed on.


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

