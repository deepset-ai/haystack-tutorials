---
layout: tutorial
featured: True
colab: https://colab.research.google.com/github/deepset-ai/haystack-tutorials/blob/main/tutorials/21_Customizing_PromptNode.ipynb
toc: True
title: "Customizing PromptNode for NLP Tasks"
level: "intermediate"
weight: 57
description: Use PromptNode and PromptTemplate for your custom NLP tasks
category: "QA"
aliases: ['/tutorials/customizing-promptnode']
download: "/downloads/21_Customizing_PromptNode.ipynb"
completion_time: False
created_at: 2023-02-16
---
    


- **Level**: Intermediate
- **Time to complete**: 20 minutes
- **Nodes Used**: `PromptNode`, `PromptTemplate`
- **Goal**: After completing this tutorial, you will have learned the basics of using PromptNode and PromptTemplates and you'll have added titles to articles from The Guardian and categorized them. 

## Overview

Use large language models (LLMs) through PromptNode and PromptTemplate to summarize and categorize your documents, and find a suitable title for them. In this tutorial, we'll use news from [The Guardian](https://www.theguardian.com/international) as documents, but you can replace them with any text you want.  

This tutorial introduces you to the basics of LLMs and PromptNode, showcases the pre-defined "summarization" template, and explains how to use PromptTemplate to generate titles for documents and categorize them with custom prompts.

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

## Trying Out PromptNode

The PromptNode is the central abstraction in Haystack's large language model (LLM) support. It uses [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base) model by default, but you can replace the default model with a flan-t5 model of a different size such as `google/flan-t5-large` or a model by OpenAI such as `text-davinci-003`.

[Large language models](https://docs.haystack.deepset.ai/docs/language_models#large-language-models-llms) are huge models trained on enormous amounts of data. That’s why these models have general knowledge of the world, so you can ask them anything and they will be able to answer.

As a warm-up, let's initialize PromptNode and see what it can do when run stand-alone: 

1. Initialize a PromptNode instance with [`google/flan-t5-large`](https://huggingface.co/google/flan-t5-large):


```python
from haystack.nodes import PromptNode

prompt_node = PromptNode(model_name_or_path="google/flan-t5-large")
```

> Note: To use PromptNode with an OpenAI model, change the model name and provide an `api_key`: 
> ```python
> prompt_node = PromptNode(model_name_or_path="text-davinci-003", api_key=<YOUR_API_KEY>)
> ```

2. Ask any general question that comes to your mind, for example:


```python
prompt_node("What is the capital of Germany?")
```


```python
prompt_node("What is the highest mountain?")
```

As `google/flan-t5-large` was trained on school math problems dataset named [GSM8K](https://huggingface.co/datasets/gsm8k) you can also ask some basic math questions:


```python
prompt_node("If Bob is 20 and Sara is 11, who is older?")
```

Now that you've initialized PromptNode and saw how it works, let's see how we can use it for more advanced tasks.

## Summarizing Documents with PromptNode

PromptNode comes with out-of-the-box prompt templates that can perform multiple tasks, such as summarization, question answering, question generation, and more. To use a templates, just provide its name to the PromptNode. 

For this task, we'll use the summarization template and news from The Guardian. Let's see how to do it.


1. Define news to use as `documents` for the PromptNode. We'll use these documents throughout the whole tutorial.


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

2. List pre-defined templates using the `get_prompt_template_names()` method. All templates come with the prompts needed to perform these tasks. 


```python
prompt_node.get_prompt_template_names()
```

3. Use the `summarization` template to generate a summary for each piece of news:


```python
prompt_node.prompt(prompt_template="summarization", documents=news)
```

Here you go! You have generated summaries of your news articles. But we're missing titles for them. Let's see how PromptNode can help us there.

## Generating Titles for News Articles with a Custom Template

The biggest benefit of PromptNode is its versatility. You can use it to perform practically any NLP task if you define your own prompt templates for them. By creating your prompt templates, you can extend the model's capabilities and use it for a broader range of NLP tasks in Haystack. 

You can define custom templates for each NLP task and register them with PromptNode. Let's create a custom template to generate descriptive titles for news:

1. Initialize a `PromptTemplate` instance. Give your template a `name` and define the prompt in `prompt_text`. To define any parameters for the prompt, add them to the `prompt_text` preceded by the `$` sign. We need a template to generate titles for our news articles. We'll call it `give-a-title`. The only parameter we need is `$news`, so let's add it to the `prompt_text`:


```python
from haystack.nodes import PromptTemplate

title_generator = PromptTemplate(
    name="give-a-title",
    prompt_text="Provide a short, descriptive title for the given piece of news. News: $news; Title:",
)
```

2. To use the new template, pass `title_generator` as the `prompt_template` to the `prompt()` method:




```python
prompt_node.prompt(prompt_template=title_generator, news=news)
```

> Note: To add a custom template to the template list, call `add_prompt_template()` with the `PromptTemplate` object pass the template contents to it. Once you do this, the next time you want to use this template, just call its name: 
> ```python
> prompt_node.add_prompt_template(PromptTemplate(name="give-a-title", prompt_text="Provide a short, descriptive title for the given piece of news. News: $news; Title:"))
> prompt_node.prompt(prompt_template="give-a-title", news=news)
> ```

There you go! You should have the titles for your news articles ready. Let's now categorize them.

## Categorizing Documents with PromptNode

You can customize PromptTemplates as much as you need. Let's try to create a template to categorize the news articles. 

1. Create another PromptTemplate called `categorize-news`. In the `prompt_text`, define the `$news` parameter, specify the categories you want to use, and ask the model not to categorize the news if it doesn't fit in the provided category list: 


```python
news_categorizer = PromptTemplate(
    name="categorize-news",
    prompt_text="Given the categories: sport, economics, culture; classify the news: $news. Only pick a category from the list, otherwise say: no suitable category",
)
```

2. Run the `prompt()` method with the `news_categorizer` template:


```python
prompt_node.prompt(prompt_template=news_categorizer, news=news)
```

Congratulations! You've summarized your documents, generated titles for them, and put them into categories, all using custom prompt templates. 

## About us


This [Haystack](https://github.com/deepset-ai/haystack/) notebook was made with love by [deepset](https://deepset.ai/) in Berlin, Germany

We bring NLP to the industry via open source!  
Our focus: Industry specific language models & large scale QA systems.  
  
Some of our other work: 
- [German BERT](https://deepset.ai/german-bert)
- [GermanQuAD and GermanDPR](https://deepset.ai/germanquad)

Get in touch:
[Twitter](https://twitter.com/deepset_ai) | [LinkedIn](https://www.linkedin.com/company/deepset-ai/) | [Discord](https://discord.com/invite/VBpFzsgRVF) | [GitHub Discussions](https://github.com/deepset-ai/haystack/discussions) | [Haystack Website](https://deepset.ai)

By the way: [we're hiring!](https://www.deepset.ai/jobs)

