---
layout: tutorial
featured: False
colab: https://colab.research.google.com/github/deepset-ai/haystack-tutorials/blob/main/tutorials/02_Finetune_a_model_on_your_data.ipynb
toc: True
title: "Fine-Tuning a Model on Your Own Data"
last_updated: 2023-02-15
level: "intermediate"
weight: 50
description: Improve the performance of your Reader by performing fine-tuning.
category: "QA"
aliases: ['/tutorials/fine-tuning-a-model']
download: "/downloads/02_Finetune_a_model_on_your_data.ipynb"
completion_time: 15 min
created_at: 2021-08-12
---
    


- **Level**: Intermediate
- **Time to complete**: 15 minutes
- **Nodes Used**: `FARMReader`
- **Goal**: After completing this tutorial, you will have learned how to fine-tune a pretrained Reader model with your own data.

## Overview

For many use cases it is sufficient to just use one of the existing public models that were trained on SQuAD or other public QA datasets (e.g. Natural Questions).
However, if you have domain-specific questions, fine-tuning your model on custom examples will very likely boost your performance.
While this varies by domain, we saw that ~ 2000 examples can easily increase performance by +5-20%.



## Preparing the Colab Environment

- [Enable GPU Runtime in Colab](https://docs.haystack.deepset.ai/docs/enabling-gpu-acceleration#enabling-the-gpu-in-colab)
- [Set logging level to INFO](https://docs.haystack.deepset.ai/docs/log-level)


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


## Create Training Data

There are two ways to generate training data

1. **Annotation**: You can use the [annotation tool](https://haystack.deepset.ai/guides/annotation) to label your data, i.e. highlighting answers to your questions in a document. The tool supports structuring your workflow with organizations, projects, and users. The labels can be exported in SQuAD format that is compatible for training with Haystack.

![Snapshot of the annotation tool](https://raw.githubusercontent.com/deepset-ai/haystack/main/docs/img/annotation_tool.png)

2. **Feedback**: For production systems, you can collect training data from direct user feedback via Haystack's [REST API interface](https://github.com/deepset-ai/haystack#rest-api). This includes a customizable user feedback API for providing feedback on the answer returned by the API. The API provides a feedback export endpoint to obtain the feedback data for fine-tuning your model further.


## Fine-tune your model

Once you have collected training data, you can fine-tune your base model.
We initialize a reader as a base model and fine-tune it on our own custom dataset (should be in SQuAD-like format).
We recommend using a base model that was trained on SQuAD or a similar QA dataset before to benefit from Transfer Learning effects.

**Recommendation**: Run training on a GPU.
If you are using Colab: Enable this in the menu "Runtime" > "Change Runtime type" > Select "GPU" in dropdown.
Then change the `use_gpu` arguments below to `True`

1. Initialize a `Reader` with the model you would like to finetune


```python
from haystack.nodes import FARMReader

reader = FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad", use_gpu=True, devices=["mps"])
```

2. Get SQuAD style data to be used for training. Below, you can fetch a dataset that we have already prepared.


```python
from haystack.utils import fetch_archive_from_http

data_dir = "data/fine-tuning"


fetch_archive_from_http(
    url="https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/squad20.tar.gz",
    output_dir=data_dir
)
```

3. Train the model with yor won data and save it to "my_model"


```python

reader.train(data_dir=data_dir, train_filename="squad20/dev-v2.0.json", use_gpu=True, n_epochs=1, save_dir="my_model")
```

4. Now initialize a new reader with your fine-tuned model


```python
new_reader = FARMReader(model_name_or_path="my_model")
```

5. Finaly, try using the `new_reader` that was initialized with your fine-tuned model.


```python
from haystack.schema import Document

new_reader.predict(query="What is the capital of Germany?", documents=[Document(content="The capital of Germany is Berlin")])
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
