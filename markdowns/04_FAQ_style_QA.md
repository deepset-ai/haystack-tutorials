---
layout: tutorial
colab: https://colab.research.google.com/github/deepset-ai/haystack-tutorials/blob/main/tutorials/04_FAQ_style_QA.ipynb
toc: True
title: "Utilizing Existing FAQs for Question Answering"
last_updated: 2023-02-02
level: "beginner"
weight: 20
description: Create a smarter way to answer new questions using your existing FAQ documents.
category: "QA"
aliases: ['/tutorials/existing-faqs']
download: "/downloads/04_FAQ_style_QA.ipynb"
---
    


While *extractive Question Answering* works on pure texts and is therefore more generalizable, there's also a common alternative that utilizes existing FAQ data.

**Pros**:

- Very fast at inference time
- Utilize existing FAQ data
- Quite good control over answers

**Cons**:

- Generalizability: We can only answer questions that are similar to existing ones in FAQ

In some use cases, a combination of extractive QA and FAQ-style can also be an interesting option.

### Prepare environment

#### Colab: Enable the GPU runtime
Make sure you enable the GPU runtime to experience decent speed in this tutorial.
**Runtime -> Change Runtime type -> Hardware accelerator -> GPU**

<img src="https://github.com/deepset-ai/haystack-tutorials/raw/main/tutorials/img/colab_gpu_runtime.jpg">

You can double check whether the GPU runtime is enabled with the following command:


```bash
%%bash

nvidia-smi
```

To start, install the latest release of Haystack with `pip`:


```bash
%%bash

pip install --upgrade pip
pip install git+https://github.com/deepset-ai/haystack.git#egg=farm-haystack[colab]
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

```


```bash
%%bash --bg

sudo -u daemon -- elasticsearch-7.9.2/bin/elasticsearch
```

### Init the DocumentStore
In contrast to Tutorial 1 (Build your first QA system), we:

* specify the name of our `embedding_field` in Elasticsearch where we'll store the embedding of our question and that is used later for calculating our similarity to the incoming user question
* set `excluded_meta_data=["question_emb"]` so that we don't return the huge embedding vectors in our search results


```python
import os
import time

from haystack.document_stores import ElasticsearchDocumentStore

# Wait 30 seconds only to be sure Elasticsearch is ready before continuing
time.sleep(30)

# Get the host where Elasticsearch is running, default to localhost
host = os.environ.get("ELASTICSEARCH_HOST", "localhost")

document_store = ElasticsearchDocumentStore(
    host=host,
    username="",
    password="",
    index="document",
    embedding_field="question_emb",
    embedding_dim=384,
    excluded_meta_data=["question_emb"],
    similarity="cosine",
)
```

### Create a Retriever using embeddings
Instead of retrieving via Elasticsearch's plain BM25, we want to use vector similarity of the questions (user question vs. FAQ ones).
We can use the `EmbeddingRetriever` for this purpose and specify a model that we use for the embeddings.


```python
from haystack.nodes import EmbeddingRetriever

retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    use_gpu=True,
    scale_score=False,
)
```

### Prepare & Index FAQ data
We create a pandas dataframe containing some FAQ data (i.e curated pairs of question + answer) and index those in elasticsearch.
Here: We download some question-answer pairs related to COVID-19


```python
import pandas as pd

from haystack.utils import fetch_archive_from_http


# Download
doc_dir = "data/tutorial4"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/small_faq_covid.csv.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

# Get dataframe with columns "question", "answer" and some custom metadata
df = pd.read_csv(f"{doc_dir}/small_faq_covid.csv")
# Minimal cleaning
df.fillna(value="", inplace=True)
df["question"] = df["question"].apply(lambda x: x.strip())
print(df.head())

# Get embeddings for our questions from the FAQs
questions = list(df["question"].values)
df["question_emb"] = retriever.embed_queries(queries=questions).tolist()
df = df.rename(columns={"question": "content"})

# Convert Dataframe to list of dicts and index them in our DocumentStore
docs_to_index = df.to_dict(orient="records")
document_store.write_documents(docs_to_index)
```

### Ask questions
Initialize a Pipeline (this time without a reader) and ask questions


```python
from haystack.pipelines import FAQPipeline

pipe = FAQPipeline(retriever=retriever)
```


```python
from haystack.utils import print_answers

prediction = pipe.run(query="How is the virus spreading?", params={"Retriever": {"top_k": 10}})
print_answers(prediction, details="medium")
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
