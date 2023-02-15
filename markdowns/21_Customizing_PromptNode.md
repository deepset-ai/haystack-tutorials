---
layout: tutorial
featured: True
colab: https://colab.research.google.com/github/deepset-ai/haystack-tutorials/blob/main/tutorials/21_Customizing_PromptNode.ipynb
toc: True
title: "Customizing PromptNode for NLP Tasks"
last_updated: 2023-02-15
level: "intermediate"
weight: 57
description: Use PromptNode and PromptTemplate for your custom NLP tasks
category: "QA"
aliases: ['/tutorials/customizing-promptnode']
download: "/downloads/21_Customizing_PromptNode.ipynb"
completion_time: 20 min
---
    


- **Level**: Intermediate
- **Time to complete**: 20 minutes
- **Nodes Used**: `PromptNode`, `PromptTemplate`
- **Goal**: After completing this tutorial, you will have learned about how to use PromptNode and PromptTemplate for your custom NLP tasks.

## Overview

Learn how to summarize, categorize your documents and find a suitable title for them with large language models using PromptNode and PromptTemplate. In this tutorial, we'll use news from [The Guardian](https://www.theguardian.com/international) as documents but you can use any text you want.  

This tutorial will introduce you to the basics of LLMs and PromptNode, showcase the pre-defined "summarization" template and explain how to use PromptTemplate to find titles for documents and categorize them with custom prompts.

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

The PromptNode is the central abstraction in Haystack's large language model (LLM) support. It uses [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base) model by default, but you can replace the default model with a flan-t5 model of a different size such as `google/flan-t5-large` or a model by OpenAI such as `text-davinci-003`.

Large language models are huge models trained on enormous amounts of data. That’s why these models have general knowledge of the world, so you can ask them anything and they will be able to answer. Let's initialize a PromptNode and see how we can prompt large language models:

1. Initialize a PromptNode instance. For this tutorial, we'll use [`google/flan-t5-large`](https://huggingface.co/google/flan-t5-large) as our large language model.


```python
from haystack.nodes.prompt import PromptNode

prompt_node = PromptNode(model_name_or_path="google/flan-t5-large")
```

> If you want to use PromptNode with an OpenAI model, change the model name and provide an `api_key`. 
```python
prompt_node = PromptNode(model_name_or_path="text-davinci-003", api_key=<YOUR_API_KEY>)
```

2. Ask a question. `google/flan-t5-large` can answer general questions.


```python
prompt_node("What is the capital of Germany?")
```


```python
prompt_node("What is the highest mountain?")
```

3. The `google/flan-t5-large` was trained on school math word problems dataset named [GSM8K](https://huggingface.co/datasets/gsm8k). That's why this model can answer basic math questions.


```python
prompt_node("If Bob is 20 and Sara is 11, who is older?")
```

Let's see how we can use PromptNode for more advanced tasks.

## Using PromptNode With a Template

PromptNode comes with out-of-the-box prompt templates to quickly interact with the large language model. These templates can perform multiple tasks, such as summarization, question answering, question generation, and more, using a single, unified model within the Haystack framework. To use these templates, just provide the documents and additional necessary information  the PromptNode. 

Let's see how we can use PromptNode to generate summary of news.


1. Define news to use as `documents` for the PromptNode. We'll use these documents for the rest of the tutorial.


```python
from haystack.schema import Document

# https://www.theguardian.com/business/2023/feb/12/inflation-may-have-peaked-but-the-cost-of-living-pain-is-far-from-over
news_economics = Document(
    """At long last, Britain’s annual inflation rate is on the way down. After hitting the highest level since the 1980s, heaping pressure on millions of households as living costs soared, official figures this week could bring some rare good news.
City economists expect UK inflation to have cooled for a third month running in January – the exact number is announced on Wednesday – helped by falling petrol prices and a broader decline in the global price of oil and gas in recent months. The hope now is for a sustained decline in the months ahead, continuing a steady drop from the peak of 11.1% seen in October.
The message from the Bank of England has been clear. Inflation is on track for a “rapid” decline over the coming months, raising hopes that the worst of Britain’s cost of living crisis is now in the rearview mirror.
There are two good reasons for this. Energy costs are moving in the right direction, while the initial rise in wholesale oil and gas prices that followed Russia’s invasion of Ukraine in February last year will soon drop from the calculation of the annual inflation rate."""
)

# https://www.theguardian.com/science/2023/feb/13/starwatch-orions-belt-and-sirius-lead-way-to-hydras-head
news_science = Document(
    """On northern winter nights, it is so easy to be beguiled by the gloriously bright constellations of Orion, the hunter, and Taurus, the bull, that one can overlook the fainter constellations.
So this week, find the three stars of Orion’s belt, follow them down to Sirius, the brightest star in the night sky, and then look eastward until you find the faint ring of stars that makes up the head of Hydra, the water snake. The chart shows the view looking south-east from London at 8pm GMT on Monday, but the view will be similar every night this week.
Hydra is the largest of the 88 modern constellations covering an area of 1,303 square degrees. To compare, nearby Orion only covers 594 square degrees. Hydra accounts for most of its area by its length, crossing more than 100 degrees of the sky (the full moon spans half a degree).
As evening becomes night and into the early hours, the rotation of Earth causes Hydra to slither its way across the southern meridian until dawn washes it from the sky. From the southern hemisphere, the constellation is easily visible in the eastern sky by mid-evening."""
)

# https://www.theguardian.com/music/2023/jan/30/salisbury-cathedral-pipe-organ-new-life-holst-the-planets
news_culture = Document(
    """A unique performance of Gustav Holst’s masterwork The Planets – played on a magnificent pipe organ rather than by an orchestra and punctuated by poems inspired by children’s responses to the music – is to be staged in the suitably vast Salisbury Cathedral.
The idea of the community music project is to introduce more people, young and old, to the 140-year-old “Father” Willis organ, one of the treasures of the cathedral.
It is also intended to get the children who took part and the adults who will watch and listen thinking afresh about the themes Holst’s suite tackles – war, peace, joy and mysticism – which seem as relevant now as when he wrote the work a century ago.
John Challenger, the cathedral’s principal organist, said: “We have a fantastic pipe organ largely as it was when built. It’s a thrilling thing. I view it as my purpose in life to share it with as many people as possible.”
The Planets is written for a large orchestra. “Holst calls for huge instrumental forces and an unseen distant choir of sopranos and altos,” said Challenger. But he has transposed the suite for the organ, not copying the effect of the orchestral instruments but finding a new version of the suite."""
)

# https://www.theguardian.com/sport/blog/2023/feb/14/multi-million-dollar-wpl-auction-signals-huge-step-forward-for-womens-sport
news_sport = Document(
    """It was only a few days ago that members of the Australian women’s cricket team were contemplating how best to navigate the impending “distraction” of the inaugural Women’s Premier League auction, scheduled during the first week of the T20 World Cup. “It’s a little bit awkward,” captain Meg Lanning said in South Africa last week. “But it’s just trying to embrace that and understanding it’s actually a really exciting time and you actually don’t have a lot of control over most of it, so you’ve just got to wait and see.”
What a pleasant distraction it turned out to be. Lanning herself will be $192,000 richer for three weeks’ work with the Delhi Capitals. Her teammate, Ash Gardner, will earn three times that playing for the Gujarat Giants. The allrounder’s figure of $558,000 is more than Sam Kerr pockets in a season with Chelsea and more than the WNBA’s top earner, Jackie Young.
If that sounds like a watershed moment, it’s perhaps because it is. And it is not the only one this past week. The NRLW made its own wage-related headlines on Tuesday, to the effect that the next (agreed in principle) collective bargaining agreement will bring with it a $1.5m salary cap in 2027, at an average salary of $62,500. Women’s rugby, too, is making moves, with news on the weekend that Rugby Australia will begin contracting the Wallaroos."""
)

news = [news_economics, news_science, news_culture, news_sport]
```

> The token limit for `google/flan-t5-large` is 512. So, all news pieces should be shorter than the limit.

2. List pre-defined templates using `get_prompt_template_names()` method. All templates come with necessary prompts to perform these tasks. 


```python
prompt_node.get_prompt_template_names()
```

3. Use `summarization` template and generate a summary for each piece of news:


```python
prompt_node.prompt(prompt_template="summarization", documents=news)
```

Now you know how to use PromptNode. Let's see how we can create a custom template for other tasks.

## Using PromptNode with a Custom Template

The biggest benefit of PromptNode is that you can use it to define and add additional prompt templates. Defining additional prompt templates makes it possible to extend the model's capabilities and use it for a broader range of NLP tasks in Haystack. 

You can define custom templates for each NLP task and register them with PromptNode. Let's create a custom template to generate descriptive titles for news:

1. Initialize a `PromptTemplate` instance. We need `name` parameter to refer to the prompt template and `prompt_text` parameter to define the prompt. We include the parameters for the prompt in `promp_text` with a `$` sign in front of the parameter name. For the new `give-a-title` template, we only need `$news` parameter.


```python
from haystack.nodes.prompt import PromptTemplate

title_generator = PromptTemplate(
    name="give-a-title",
    prompt_text="Provide a short, descriptive title for the given piece of news. News: $news; Title:",
)
```

2. To use the new template, pass the new `title_generator` as the `prompt_template` to the `prompt()` method.




```python
prompt_node.prompt(prompt_template=title_generator, news=news)
```

> If you add a custom template to the template list, call `add_prompt_template()` with the `PromptTemplate` object and you can start using the template only with its `name`. 
```python
prompt_node.add_prompt_template(PromptTemplate(name="give-a-title", prompt_text="Provide a short, descriptive title for the given piece of news. News: $news; Title:"))
prompt_node.prompt(prompt_template="give-a-title", news=news)
```

You can customize PromptTemplates as much as you want according to your needs Let's try to categorize the news and see how you can customize the prompt further. 

1. Create another PromptTemplate called `categorize-news`. In the `prompt_text`, define `$news` parameter, give categories and ask model not to categorize the news if it doesn't fit in the provided category list: 


```python
news_categorizer = PromptTemplate(
    name="categorize-news",
    prompt_text="Given the categories: sport, economics, culture; classify the news: $news. Only pick a category from the list, otherwise say: no suitable category",
)
```

2. Run the `prompt()` method with `news_categorizer` template.


```python
prompt_node.prompt(prompt_template=news_categorizer, news=news)
```

And that's it! Now you know how to use PromptNode and create custom prompts with PromptTemplate.

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

