---
layout: tutorial
colab: https://colab.research.google.com/github/deepset-ai/haystack-tutorials/blob/main/tutorials/20_Using_Haystack_with_REST_API.ipynb
toc: True
title: "Using Haystack with REST API"
last_updated: 2022-12-08
level: "advanced"
weight: 115
description: Create a production-ready pipeline and interact with Haystack REST API.
category: "QA"
aliases: ['/tutorials/using-haystack-with-rest-api']
---
    

# Tutorial: Using Haystack with REST API

- **Level**: Advanced
- **Time to complete**: 30 minutes
- **Prerequisites**: Basic understanding of Docker, basic knowledge of Haystack pipelines 
- **Nodes Used**: `ElasticsearchDocumentStore`, `EmbeddingRetriever`
- **Goal**: Learn how you can interact with Haystack through REST API.

## Overview

Haystack enables you to apply the latest NLP technology to your own data and create production-ready applications. Building an end-to-end NLP application requires the combination of multiple concepts:
* **DocumentStore** is the component in Haystack responsible for loading and storing text data in form of [Documents](https://docs.haystack.deepset.ai/docs/documents_answers_labels#document). In this tutorial, the DocumentStore will use Elasticsearch behind the scene.
* **Haystack** pipelines convert files into Documents, index them to DocumentStore, and run NLP tasks such as question answering and document search.
* **REST API**, as a concept, allows applications to interact with each other by handling their queries and returning responses. There is `rest_api` application within Haystack that exposes Haystack's functionalities through a RESTful API.
* **Docker** simplifies the environment setup needed to have Elasticsearch and Haystack API running.

This tutorial introduces you to all the concepts needed to build an end-to-end document search application. After completing this tutorial, you will have learned how to create a pipeline YAML file, index files, and how to query your pipeline using REST API.

## Preparing the Environment

1. Install [Docker Compose](https://docs.docker.com/compose/), and launch Docker.

If you installed Docker Desktop, you just need to start the application. Run `docker info` to see if Docker is up and running.

```bash
docker info
```

2. Clone Haystack repository.

Haystack provides a `docker-compose.yml` file that defines services for Haystack API and Elasticsearch. Clone the Haystack repository to be able to run the `docker-compose.yml` file locally. 

```bash
git clone https://github.com/deepset-ai/haystack.git
```

3. Update `docker-compose.yml` file.

Find and open the `docker-compose.yml` file. In the file, there is a predefined volume under `haystack-api` service. You need to replace `<PATH_TO_HAYSTACK>` with the path to the folder where you cloned the Haystack repository. Here's an example volume:

```yaml
volumes:
  - /Users/admin/Desktop/examples/haystack/rest_api/rest_api/pipeline:/home/user/rest_api/pipeline
```

## Create Pipeline YAML File

YAML files are widely used for configurations. You can define Haystack pipelines as YAML files and then `load_from_yaml()` method loads the pipeline defined in YAML into a Python object. In a YAML file, the `components` section defines all pipeline nodes and the `pipelines` section defines how these nodes are connected to each other to form a pipeline. Let's start with defining query and indexing pipelines.

1. Create a document search pipeline.

Time to design a document search pipeline from scratch. This will be your query pipeline. Create a new file named `document-search.haystack-pipeline.yml` in the `/pipeline` folder under `/rest_api` in the Haystack code base. Then, create a `PIPELINE_YAML_PATH` variable in the `docker-compose.yml` with the new file name. `PIPELINE_YAML_PATH` variable will tell `rest_api` which YAML file to run. 

```yaml
environment:
  - DOCUMENTSTORE_PARAMS_HOST=elasticsearch
  - PIPELINE_YAML_PATH=/home/user/rest_api/pipeline/document-search.haystack-pipeline.yml
  ...
```

A document search pipeline requires a DocumentStore and a Retriever. Use `ElasticsearchDocumentStore` and `EmbeddingRetriever` for these nodes respectively and define them under `components`. For each component, set `type` to a node class in Haystack, set `name` to how you want to call this node in the pipeline YAML, and use `params` to set the node parameters. As DocumentStore parameters, provide `embedding_dim` required for the `embedding_model` and as Retriever parameters, provide `document_store`, `embedding_model`, and a `top_k` value. Let's start by defining the pipeline nodes:

```yaml
components:
  - name: DocumentStore
    type: ElasticsearchDocumentStore
    params:
      embedding_dim: 384
  - name: Retriever
    type: EmbeddingRetriever
    params:
      document_store: DocumentStore
      top_k: 5 
      embedding_model: sentence-transformers/all-MiniLM-L6-v2
```

After you define the nodes, create a query pipeline in the `pipelines` section. Here, `name` refers to the name of the pipeline, and `nodes` defines the order of the nodes in the pipeline: 

```yaml
pipelines:
  - name: query 
    nodes:
      - name: Retriever
        inputs: [Query]
```

2. Create an indexing pipeline.

You can define an indexing pipeline in the same pipeline YAML file and index your documents to Elasticsearch through `rest_api`. For that, create `FileTypeClassifier`, `TextConverter`, `PDFToTextConverter`, and `PreProcessor` nodes. For `PreProcessor`, use `params` to define how you want to split your documents: 

```yaml
components:
  ...
  - name: FileTypeClassifier
    type: FileTypeClassifier
  - name: TextFileConverter
    type: TextConverter
  - name: PDFFileConverter
    type: PDFToTextConverter
  - name: Preprocessor
    type: PreProcessor
    params:
      split_by: word
      split_length: 1000
      split_overlap: 50 
      split_respect_sentence_boundary: True 
```

Then, in the `pipelines` section of the YAML file, create a new pipeline called `indexing`. In this pipeline, indicate how the nodes you just defined are connected to each other, Retriever, and DocumentStore. This indexing pipeline supports `.TXT` and `.PDF` files and pre-processes them before loading to the Elasticsearch.

```yaml
pipelines:
  ...
  - name: indexing
    nodes:
      - name: FileTypeClassifier
        inputs: [File]
      - name: TextFileConverter
        inputs: [FileTypeClassifier.output_1]
      - name: PDFFileConverter
        inputs: [FileTypeClassifier.output_2]
      - name: Preprocessor
        inputs: [PDFFileConverter, TextFileConverter]
      - name: Retriever
        inputs: [Preprocessor]
      - name: DocumentStore
        inputs: [Retriever]
```

After completing query and indexing pipelines, add `version: ignore` to the top of the file. Now, the pipeline YAML is ready.

```yaml
version: ignore

components:
  - name: DocumentStore
    type: ElasticsearchDocumentStore
    params:
      embedding_dim: 384
  - name: Retriever
    type: EmbeddingRetriever
    params:
      document_store: DocumentStore
      top_k: 5 
      embedding_model: sentence-transformers/all-MiniLM-L6-v2
  - name: FileTypeClassifier
    type: FileTypeClassifier
  - name: TextFileConverter
    type: TextConverter
  - name: PDFFileConverter
    type: PDFToTextConverter
  - name: Preprocessor
    type: PreProcessor
    params:
      split_by: word
      split_length: 1000
      split_overlap: 50 
      split_respect_sentence_boundary: True

pipelines:
  - name: query 
    nodes:
      - name: Retriever
        inputs: [Query]
  - name: indexing
    nodes:
      - name: FileTypeClassifier
        inputs: [File]
      - name: TextFileConverter
        inputs: [FileTypeClassifier.output_1]
      - name: PDFFileConverter
        inputs: [FileTypeClassifier.output_2]
      - name: Preprocessor
        inputs: [PDFFileConverter, TextFileConverter]
      - name: Retriever
        inputs: [Preprocessor]
      - name: DocumentStore
        inputs: [Retriever]
```

Feel free to play with the pipeline setup later on. Add or remove some nodes, change the parameters, or add new ones. For more options for nodes and parameters, check out [Haystack API Reference](https://docs.haystack.deepset.ai/reference/answer-generator-api).

## Launch Haystack API and Elasticsearch

Pipelines are ready. Now, go to the directory where `docker-compose.yml` is and run `docker-compose up`. This command will install all necessary packages, set up the environment, and launch both Elasticsearch and Haystack API. 

```bash
docker-compose up
```

Before continuing, test if everything is OK with the Haystack API by sending a cURL request to the `/initialized` endpoint. You can use the command line of your computer or tools like [Postman](https://www.postman.com/) to send cURL requests. If everything works fine, you will get `true` as a response.

```bash
curl --request GET http://127.0.0.1:8000/initialized
```

## Index Files to Elasticsearch

Right now, the Elasticsearch instance is empty. Haystack API provides a `/file-upload` endpoint to upload files to Elasticsearch using the indexing pipeline defined in the pipeline YAML. After indexing files to Elasticsearch, you will be able to perform document search.

1. Download example files.

Download the [example files](https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/article_txt_countries_and_capitals.zip) you will be indexing to Elasticsearch. In the folder, there are text files about countries and capitals crawled from [Wikipedia](https://en.wikipedia.org/wiki/Category:Lists_of_countries_by_continent).

2. Index files to Elasticsearch.

You can send cURL requests to the `/file-upload` endpoint to upload files to the Elasticsearch instance. Replace `<PATH_TO_FOLDER>` with the path to the example files on your computer and send the POST request. If the file is successfully uploaded, you will get `null` as a response.

```bash
curl --request POST \
     --url http://127.0.0.1:8000/file-upload \
     --header 'accept: application/json' \
     --header 'content-type: multipart/form-data' \
     --form files=@<PATH_TO_FOLDER>0_Minsk.txt \
     --form meta=null
```

This method is not convenient for uploading multiple files to the Elasticsearch instance as replacing the file names in the request by hand is difficult. Instead, create a python file in the folder where you keep the example files, put the code below into the python file, and run the python script. This python code takes the name of every file in the folder and sends the POST request to the URL `http://127.0.0.1:8000/file-upload`. With the `print(response.text, ":", file_name)` line, you will be able to see the names of all indexed files. Make sure you see the "Completed" text before continuing.

```python
import os
import requests
 
file_list = os.listdir()
url = "http://127.0.0.1:8000/file-upload"
payload = {"meta": "null"}
headers = {"accept": "application/json"}

for file_name in file_list:
    files = {"files": (file_name, open(file_name, "rb"), "text/plain")}
    response = requests.post(url, data=payload, files=files, headers=headers)
    print(response.text, ":", file_name)

print("Completed")
```

## Voilà! Make a query!

The application is ready! Send a cURL request to retrieve documents about _"climate in Scandinavia"_. 

```bash
curl --request POST \
     --url http://127.0.0.1:8000/query \
     --header 'accept: application/json' \
     --header 'content-type: application/json' \
     --data '{
     "query": "climate in Scandinavia"
     }'
```

As a response, you will get a `QueryResponse` object consisting of `query`, `answers`, and `documents`. Documents related to your query will be under the `documents` attribute of the object.

```bash
{
  "query": "climate in Scandinavia",
  "answers": [],
  "documents": [
    {
      "id": "52937ad257317032b9aed9750b5fcbb7",
      "content": "Even though temperature patterns differ between north and south, the summer climate is surprisingly similar all through the entire country in spite of the large latitudinal differences. This is due to the south's being surrounded by a greater mass of water, with the wider Baltic Sea and the Atlantic air passing over lowland areas from the south-west. ...",
      "content_type": "text",
      "meta": {
        "_split_id": 7,
        "name": "43_Sweden.txt"
      },
      "score": 0.569930352925387
    },
    ...
  ]
}
```  

## About us

This [Haystack](https://github.com/deepset-ai/haystack/) tutorial was made with love by [deepset](https://deepset.ai/) in Berlin, Germany

We bring NLP to the industry via open source!  
Our focus: Industry specific language models & large scale QA systems.  
  
Some of our other work: 
- [German BERT](https://deepset.ai/german-bert)
- [GermanQuAD and GermanDPR](https://deepset.ai/germanquad)
- [FARM](https://github.com/deepset-ai/FARM)

Get in touch:
[Twitter](https://twitter.com/deepset_ai) | [LinkedIn](https://www.linkedin.com/company/deepset-ai/) | [Discord](https://haystack.deepset.ai/community/join) | [GitHub Discussions](https://github.com/deepset-ai/haystack/discussions) | [Website](https://deepset.ai)

By the way: [we're hiring!](https://www.deepset.ai/jobs)

