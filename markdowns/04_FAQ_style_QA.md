---
layout: tutorial
featured: False
colab: https://colab.research.google.com/github/deepset-ai/haystack-tutorials/blob/main/tutorials/04_FAQ_style_QA.ipynb
toc: True
title: "Utilizing Existing FAQs for Question Answering"
last_updated: 2023-02-17
level: "beginner"
weight: 20
description: Create a smarter way to answer new questions using your existing FAQ documents.
category: "QA"
aliases: ['/tutorials/existing-faqs']
download: "/downloads/04_FAQ_style_QA.ipynb"
completion_time: False
created_at: 2021-08-12
---
    

- **Level**: Beginner
- **Time to complete**: 15 minutes
- **Nodes Used**: `InMemoryDocumentStore`, `EmbeddingRetriever`
- **Goal**: Learn how to use the `EmbeddingRetriever` in a `FAQPipeline` to answer incoming questions by matching them to the most similar questions in your existing FAQ.

# Overview
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

### Create a simple DocumentStore
The InMemoryDocumentStore is good for quick development and prototyping. For more scalable options, check-out the [docs](https://docs.haystack.deepset.ai/docs/document_store).


```python
from haystack.document_stores import InMemoryDocumentStore

document_store = InMemoryDocumentStore()
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
We create a pandas dataframe containing some FAQ data (i.e curated pairs of question + answer) and index those in our documentstore.
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

# Create embeddings for our questions from the FAQs
# In contrast to most other search use cases, we don't create the embeddings here from the content of our documents,
# but rather from the additional text field "question" as we want to match "incoming question" <-> "stored question".
questions = list(df["question"].values)
df["embedding"] = retriever.embed_queries(queries=questions).tolist()
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

# Run any question and change top_k to see more or less answers
prediction = pipe.run(query="How is the virus spreading?", params={"Retriever": {"top_k": 1}})

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
