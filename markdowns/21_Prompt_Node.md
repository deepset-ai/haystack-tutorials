---
layout: tutorial
colab: https://colab.research.google.com/github/deepset-ai/haystack-tutorials/blob/main/tutorials/21_Prompt_Node.ipynb
toc: True
title: "The Basics of PromptNode"
last_updated: 2023-02-10
level: "beginner"
weight: 45
description: Use large language models for various NLP tasks
category: "QA"
aliases: ['/tutorials/basics-of-prompt-node']
download: "/downloads/21_Prompt_Node.ipynb"
---
    


- **Level**: Beginner
- **Time to complete**: 15 minutes
- **Nodes Used**: `PromptNode`
- **Goal**: After completing this tutorial, you will have learned about how to leverage the power of large language models with PromptNode.

## Overview

Learn various ways to use PromptNode as a stand-alone node. This tutorial will introduce you the basics of PromptNode, showcase different out-of-the-box templates and explain how to implement a template for your custom NLP task.

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

## Using PromptNode as a Stand-Alone Node

The PromptNode class is the central abstraction in Haystack's large language model (LLM) support. It uses [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base) model by default, but you can replace the default model with a flan-t5 model of a different size such as `google/flan-t5-large` or a model by OpenAI such as `text-davinci-003`.

Large language models are huge models trained on enormous amounts of data. That’s why these models have general knowledge of the world, so you can ask them anything and they will be able to answer. Let's initialize a PromptNode and ask some questions:


```python
from haystack.nodes.prompt import PromptNode

prompt_node = PromptNode(model_name_or_path="google/flan-t5-xl")
prompt_node("What is the capital of Germany?")
```

The `google/flan-t5-large` was trained on school math word problems dataset named [GSM8K](https://huggingface.co/datasets/gsm8k). That's why this model can successfully answer math questions.


```python
prompt_node("If Bob is 20 and Sara is 21, who is older?")
```

> To use `PromptNode` with an OpenAI model, just provide an `api_key` and the model name you want to use: `prompt_node = PromptNode(model_name_or_path="text-davinci-003", api_key=<YOUR_API_KEY>)`
```python
prompt_node = PromptNode(model_name_or_path="text-davinci-003", api_key=<YOUR_API_KEY>)
```

## Using PromptNode With a Template

You can also use PromptNode with out-of-the-box templates. These templates can perform multiple tasks, such as summarization, question answering, question generation, and more, using a single, unified model within the Haystack framework. 

The `get_prompt_template_names()` method lists pre-defined templates:


```python
prompt_node.get_prompt_template_names()
```


```python
# https://www.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/documents/jpmc-esg-report-2020.pdf
esg_content = """Responsible resource and waste management are important
elements of our sustainability strategy, helping us reduce our
impacts while improving efficiency and reducing costs. Our
focus is on reducing our water and waste footprint, coupled with
responsible disposal of the waste we produce. To drive progress
in these areas, in 2021 we set new targets to reduce our water
consumption by 20% by 2030 and internal paper use by 90% by
2025 compared with 2017 baselines. And we have also committed
that by the end of 2021 we will source 100% of our paper from
certified sources, meaning the products come from responsibly
managed forests that provide environmental, social and economic
benefits. We work to recycle paper, as well as non-paper waste,
throughout our buildings and branches where recycling services
are available and economically feasible. We are working to
optimize existing recycling services, expand such services to new
locations and explore opportunities to bring composting services
to more of our corporate locations with cafeterias. We also
carefully select vendors to dispose our e-waste responsibly, with
100% diverted from landfills.
We recognize that the environmental and social impact of our
operations extends to our suppliers. As such, we seek to do
business with suppliers that share our values and commitment to
making a positive impact in the communities where we operate.
We encourage our suppliers to develop internal programs, as
well as targets, to foster a culture of sustainability. We expect
them to conduct their operations in a manner that protects the
environment by making reasonable efforts to meet industry best
practices and standards with respect to the reduction of energy
use, GHG emissions, waste and water use."""
```


```python
prompt_node.prompt("question-generation", documents=[esg_content])
```


```python
prompt_node.prompt("summarization", documents=[esg_content])
```

## Using PromptNode with a Custom Template

One of the benefits of PromptNode is that you can use it to define and add additional prompt templates the model supports. Defining additional prompt templates makes it possible to extend the model's capabilities and use it for a broader range of NLP tasks in Haystack. 

Prompt engineers can define templates for each NLP task and register them with PromptNode. For a custom template, create a `PromptTemplate` with `name` and `prompt_text` parameters. You need to define necessary parameters for the prompt in `promp_text` with `$` sign. Let's create a custom template to generate a TL;DR section for the ESG report:


```python
from haystack.nodes.prompt import PromptTemplate

tldr_generator = PromptTemplate(
    name="tldr-generator",
    prompt_text="Please provide a interesting TLDR section of the given document. Documents: $documents; TLDR:",
)
```


```python
prompt_node.prompt(tldr_generator, documents=[esg_content])
```

You can also add the new template to the template list with the `add_prompt_template()` and use it by its name. Create another PromptTemplate and pass it to `add_prompt_template()` as argument to be able to call with `prompt()` method:


```python
prompt_node.add_prompt_template(
    PromptTemplate(
        name="give-a-title",
        prompt_text="Provide short titles for the following documents. Documents: $documents; Title,",
    )
)
```


```python
prompt_node.prompt(prompt_template="give-a-title", documents=[esg_content])
```

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

