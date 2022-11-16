---
layout: tutorial
colab: https://colab.research.google.com/github/deepset-ai/haystack-tutorials/blob/main/tutorials/21_distill_a_reader.ipynb
toc: True
title: "Distill a Reader"
last_updated: 2022-11-16
level: "intermediate"
weight: 115
description: Transfer a Reader's question answering ability to a smaller, more efficient model.
category: "QA"
aliases: ['/tutorials/distill-reader']
---
    

# Tutorial: Distill a Reader

- **Level**: Advanced
- **Time to complete**: 30 minutes
- **Nodes Used**: `FARMReader`
- **Goal**: Distil the question answering capabilities of the larger BERT base Reader model into a smaller TinyBERT Reader model.


## Overview

Model distillation is the process of teaching a smaller model to imitate the performance of a larger, better trained model. By distilling one model into another, you end up with a more computationally efficient version of the original with only a slight trade-off in accuracy. In this tutorial, you will learn how to perform one form of model distillation on Reader models in Haystack. Model distillation is a complex topic and an active area of research so if want to learn more about it, see [Model Distillation](https://docs.haystack.deepset.ai/docs/model_distillation).

## Preparing the Colab Environment

<details>
- [Enable GPU Runtime in GPU](https://docs.haystack.deepset.ai/docs/enable-gpu-runtime-in-colab)
- [Check if GPU is Enabled](https://docs.haystack.deepset.ai/docs/check-if-gpu-is-enabled)
- [Set logging level to INFO](https://docs.haystack.deepset.ai/docs/set-the-logging-level)
</details>


## Installing Haystack

To start, let's install the latest release of Haystack with `pip`:


```bash
%%bash

pip install --upgrade pip
pip install farm-haystack[colab]
```

## Augmenting Training Data

Having more human annotated training data is useful at all levels of model training. However, intermediate layer distillation can benefit even from synthetically generated data, since it is a less exact type of training. In this tutorial, we'll be using the [`augment_squad.py` script](https://github.com/deepset-ai/haystack/blob/main/haystack/utils/augment_squad.py) to augment our dataset. It creates artificial copies of question answering samples by replacing randomly chosen words with words of similar meaning. This meaning similarity is determined by their vector representations in a GLoVe word embedding model.

1. Download the `augment_squad.py` script.


```python
!wget https://raw.githubusercontent.com/deepset-ai/haystack/main/haystack/utils/augment_squad.py
```

2. Download a small slice of the SQuAD question answering database.


```python
from haystack.utils import fetch_archive_from_http

doc_dir = "data/distil_a_reader"
squad_dir = doc_dir + "/squad"

s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/squad_small.json.zip"
fetch_archive_from_http(url=s3_url, output_dir=squad_dir)
```

 3. Download a set of GLoVe vectors.


```python
glove_dir = doc_dir + "/glove"

glove_url = "https://nlp.stanford.edu/data/glove.6B.zip"
fetch_archive_from_http(url=glove_url, output_dir=glove_dir)
```

This tutorial uses a smaller set of vectors and a smaller dataset to make it faster. For real use cases, pick larger versions of both.

4. Run the `augment_squad.py` script to create an augmented dataset.


```python
!python augment_squad.py \
    --squad_path data/distil_a_reader/squad/squad_small.json \
    --glove_path data/distil_a_reader/glove/glove.6B.300d.txt \
    --output_path augmented_dataset.json \
    --multiplication_factor 2
```

The multiplication factor determines how many augmented samples we're generating. Setting it to 2 makes it much quicker to run. In real use cases, set this to something like 20.

## Distilling a Reader

Distillation in Haystack is done in two phases:
- Intermediate layer distillation ensures that the teacher and student models behave similarly. This can be performed using the augmented data. While intermediate layer distillation is optional, it will improve the performance of the model after training.
- Prediction layer distillation optimizes the model for the specific task. This must be performed using the non-augmented data.


1. Initialize the teacher model.


```python
from haystack.nodes import FARMReader

teacher = FARMReader(model_name_or_path="deepset/bert-base-uncased-squad2", use_gpu=True)
```

Here we are using [`deepset/bert-base-uncased-squad2`](https://huggingface.co/deepset/bert-base-uncased-squad2), a base sized BERT model trained on SQuAD.

2. Initialize the student model.


```python
student = FARMReader(model_name_or_path="huawei-noah/TinyBERT_General_6L_768D", use_gpu=True)
```

Here we are using a TinyBERT model that is smaller than the teacher model. You can pick any other student model, so long as it uses the same tokenizer as the teacher model. Also, the number of layers in the teacher model must be a multiple of the number of layers in the student.

3. Perform intermediate layer distillation.


```python
student.distil_intermediate_layers_from(teacher, data_dir=".", train_filename="augmented_dataset.json", use_gpu=True)
```

4. Perform prediction layer distillation.


```python
student.distil_prediction_layer_from(teacher, data_dir="data/squad20", train_filename="dev-v2.0.json", use_gpu=True)
```

5. Save the student model.


```python
student.save(directory="my_distilled_model")
```



# Next Steps

To learn how to measure the performance of these Reader models, see [Evaluate a Reader model](https://haystack.deepset.ai/tutorials/05_evaluate_a_reader).

## About us

This [Haystack](https://github.com/deepset-ai/haystack/) notebook was made with love by [deepset](https://deepset.ai/) in Berlin, Germany

We bring NLP to the industry via open source!  
Our focus: Industry specific language models & large scale QA systems.  
  
Some of our other work: 
- [German BERT](https://deepset.ai/german-bert)
- [GermanQuAD and GermanDPR](https://deepset.ai/germanquad)

Get in touch:
[Twitter](https://twitter.com/deepset_ai) | [LinkedIn](https://www.linkedin.com/company/deepset-ai/) | [Discord](https://haystack.deepset.ai/community/join) | [GitHub Discussions](https://github.com/deepset-ai/haystack/discussions) | [Website](https://deepset.ai)

By the way: [we're hiring!](https://www.deepset.ai/jobs)
