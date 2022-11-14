---
layout: tutorial
colab: https://colab.research.google.com/github/deepset-ai/haystack-tutorials/blob/main/tutorials/03_finetune_a_reader.ipynb
toc: True
title: "Fine-Tune a Reader"
last_updated: 2022-11-14
level: "intermediate"
weight: 50
description: Improve the performance of your Reader by performing fine-tuning.
category: "QA"
aliases: ['/tutorials/fine-tuning-a-model', '/tutorials/02_Finetune_a_model_on_your_data.ipynb', '/tutorials/fine-tune-reader']
---
    

# Fine-Tune a Reader

- **Level**: Intermediate
- **Time to complete**: 20 minutes
- **Nodes Used**: `FARMReader`
- **Goal**: Learn how to improve the performance of a Reader model by performing fine-tuning.

## Overview

Fine-tuning can improve your Reader's performance on question answering, especially if you're working with very specific domains. While many of the existing public models trained public question answering datasets are enough for most use cases, fine-tuning can help your model understand the phrases and terms specific to your field. While this varies for each domain and dataset, we've had cases where ~2000 examples increased performance by as much as +5-20%. After completing this tutorial, you will have all the tools needed to fine-tune a pretrained model on your own dataset.

## Preparing the Colab Environment

- [Enable GPU Runtime in GPU](https://docs.haystack.deepset.ai/docs/enable-gpu-runtime-in-colab)
- [Check if GPU is Enabled](https://docs.haystack.deepset.ai/docs/check-if-gpu-is-enabled)
- [Set logging level to INFO](https://docs.haystack.deepset.ai/docs/set-the-logging-level)


## Installing Haystack

To start, let's install the latest release of Haystack with `pip`:


```bash
%%bash

pip install --upgrade pip
pip install farm-haystack[colab]
```


## Creating Training Data

To start fine-tuning your Reader model, you need question answering data in the [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) format. One sample from this data should contain a question, a text answer, and the document in which this answer can be found.

You can start generating your own training data using one of the two tools that we offer:

1. **Annotation Tool**: You can use the deepset [Annotation Tool](https://haystack.deepset.ai/guides/annotation) to write questions and highlight answers in a document. The tool supports structuring your workflow with organizations, projects, and users. You can then export the question-answer pairs in the SQuAD format that is compatible with fine-tuning in Haystack.

2. **Feedback Mechanism**: In a production system, you can collect users' feedback to model predictions with Haystack's [REST API interface](https://github.com/deepset-ai/haystack#rest-api) and use this as training data. To learn how to interact with the user feedback endpoints, see [User Feedback](https://docs.haystack.deepset.ai/docs/domain_adaptation#user-feedback).



## Fine-tuning the Reader

1. Initialize the Reader, supplying the name of the base model you wish to improve.


```python
from haystack.nodes import FARMReader

reader = FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad", use_gpu=True)
```

We recommend using a model that was trained on SQuAD or a similar question answering dataset to benefit from transfer learning effects. In this tutorial, we are using [distilbert-base-uncased-distilled-squad](https://huggingface.co/distilbert-base-uncased-distilled-squadbase), a base-sized DistilBERT model that was trained on SQuAD. To learn more about what model works best for your use case, see [Models](https://haystack.deepset.ai/pipeline_nodes/reader#models).

2. Provide the SQuAD format training data to the `Reader.train()` method.


```python
data_dir = "data/squad20"
reader.train(
    data_dir=data_dir,
    train_filename="dev-v2.0.json",
    use_gpu=True,
    n_epochs=1,
    save_dir="my_model"
)
```

With the default parameters above, we are starting with a base model trained on the SQuAD training dataset and we are further fine-tuning it on the SQuAD development dataset. To fine-tune the model for your domain, replace `train_filename` with your domain-specific dataset.

To perform evaluation over the course of fine-tuning, see [FARMReader.train() API](https://docs.haystack.deepset.ai/reference/reader-api#farmreadertrain) for the relevant arguments.

## Saving and Loading

The model is automatically saved at the end of fine-tuning in the `save_dir` that you specified.
However, you can also manually save the Reader again by running:


```python
reader.save(directory="my_model")
```

To load a saved model, run:


```python
new_reader = FARMReader(model_name_or_path="my_model")
```

# Next Steps

Now that you have a model with improved performance, why not transfer its question answering capabilities into a smaller, faster model? Starting with this new model, you can use model distillation to create a more efficient model with only a slight tradeoff in performance. To learn more, see [Distil a Reader](https://haystack.deepset.ai/tutorials/04_distil_a_reader).

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
