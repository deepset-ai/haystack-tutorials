---
layout: tutorial
colab: https://colab.research.google.com/github/deepset-ai/haystack-tutorials/blob/main/tutorials/20_REST_API.ipynb
toc: True
title: "Haystack with REST API"
last_updated: 2022-11-23
level: "intermediate"
weight: 67
description: Create a production-ready pipeline and interact with REST API
category: "QA"
aliases: ['/tutorials/rest-api']
---
    

# Tutorial: Using Haystack with REST API

- **Level**: Advanced
- **Time to complete**: 30 minutes
- **Prerequisites**: Basic understanding of Docker, basic knowledge of Haystack pipelines 
- **Nodes Used**: `ElasticsearchDocumentStore`, `EmbeddingRetriever`
- **Goal**: Learn how you can interact with Haystack through REST API.

## Overview

Haystack enables you to apply the latest NLP technology to your own data and create production-ready applications. Building an end-to-end NLP application requires the combination of multiple concepts. Here are those consepts:
* **DocumentStore** stores the data. You will use Elasticsearch for this tutorial.
* **Haystack** pipelines convert documents, index them to DocumentStore, and run NLP tasks such as question answering and document search.
* **REST API** interacts with Haystack and other applications to pass query and response between them.
* **Docker** simplifies the environment set-up needed to have Elasticsearch running.

This tutorial introduces you to all the concepts needed to build an end-to-end document search application. After completing this tutorial, you will have learned how to create a pipeline YAML file, index files and query your application using REST API.

## Preparing the Environment

1. Update or install Docker and Docker Compose, then launch Docker:

Set up Docker to start an Elasticsearch container. 


```bash
%%bash

apt-get update && apt-get install docker && apt-get install docker-compose
service docker start
```

2. Install Haystack:

Install the latest version of Haystack from the main branch and all its dependencies. REST API and `xpdf` are also required.  


```bash
%%bash 

pip install --upgrade pip
pip install 'farm-haystack[all]'
pip install -e rest_api/

brew install xpdf # required for PDFToTextConverter node
```

3. Clone Haystack repository:

Haystack provides a `docker-compose.yml` file that defines a container for Elasticsearch. Clone the Haystack repository to be able to run the `docker-compose.yml` file locally. 


```bash
%%bash

git clone https://github.com/deepset-ai/haystack.git
```

4. Update the `docker-compose.yml` file:

The current `docker-compose.yml` file has basic settings and it also provides an image for Elasticsearch container which needs to change. For that, go to `docker-compose.yml` and convert `image: "deepset/elasticsearch-countries-and-capitals"` line into `image: "docker.elastic.co/elasticsearch/elasticsearch:7.9.2"`. The new image is going to provide an empty Elasticsearch instance instead of an instance with some indexed articles about countries and capital cities. 


```yaml
services:
  ...
  elasticsearch:
    image: "docker.elastic.co/elasticsearch/elasticsearch:7.9.2"
    ports:
      - 9200:9200
    restart: on-failure
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms1024m -Xmx1024m"
    healthcheck:
        test: curl --fail http://localhost:9200/_cat/health || exit 1
        interval: 10s
        timeout: 1s
        retries: 10
  ...
```

5. Launch Elasticsearch:

Go to the directory where `docker-compose.yml` is and start the Elasticsearch container. You might have realised that there are settings for haystack and ui containers provided in the `docker-compose.yml`. They are not necessary for in this tutorial. Run `docker-compose up` with `elasticsearch` argument so that only the Elasticsearch container launches.


```bash
%%bash

cd haystack
docker-compose up elasticsearch
```

6. Check Elasticsearch health

Launching Elasticsearch takes some time, so, make sure you have a `healthy` Elasticsearch container before continuing. You can check the container health with the `docker ps` command. A `healthy` container should have a healthy status on port 9200.

![command line output of `docker ps`](https://github.com/deepset-ai/haystack-tutorials/blob/main/tutorials/img/tutorial20_elasticsearch_healthy.png?raw=true)


```bash
%%bash

docker ps
```

## Create Pipeline YAML File

YAML files are widely used for confugurations. Haystack enables defining pipelines as YAML files and `load_from_yaml` method loads pipelines from YAML file. In a YAML file, `components` section defines all pipeline nodes and `pipelines` section defines how these nodes are connected to each other to form a pipeline. Let's start with defining query and indexing pipelines.

1. Create a Document Search Pipeline

As the query pipeline, you will create a Document Search pipeline from scratch. Create a new file named `document-search.haystack-pipeline.yml` in `/pipeline` folder under `/rest_api` in Haystack code base. Then, update `PIPELINE_YAML_PATH` value in `rest_api/config.py` with the new file name. `PIPELINE_YAML_PATH` will tell the REST API which YAML file to run. 

```python
PIPELINE_YAML_PATH = os.getenv(
    "PIPELINE_YAML_PATH", str((Path(__file__).parent / "pipeline" / "document-search.haystack-pipeline.yml").absolute())
)
```

A document search pipeline requires a DocumentStore and a Retriever. Use `ElasticsearchDocumentStore` and `EmbeddingRetriever` for these nodes respectively and define them under `components`. For each component, `type` needs to refer to a pipeline node in Haystack, `name` refers how these nodes are called in the pipeline YAML, and `params` is used for the parameters that can be provided. As Retriever parameters, provide `document_store`, `embedding_model` and a `top_k` value. 

```yaml
components:
  - name: DocumentStore
    type: ElasticsearchDocumentStore
  - name: Retriever
    type: EmbeddingRetriever
    params:
      document_store: DocumentStore
      top_k: 5 
      embedding_model: sentence-transformers/multi-qa-mpnet-base-dot-v1
```

After defining components, create a query pipeline in `pipelines` section. Here, `name` refers to the name of the pipeline and `nodes` defines how the pipeline is built. 

```yaml
pipelines:
  - name: query 
    nodes:
      - name: Retriever
        inputs: [Query]
```

2. Create an Indexing Pipeline

You can define an indexing pipeline in the same pipeline YAML file and index your documents to Elasticsearch through REST API. For that, create `FileTypeClassifier`, `TextConverter`, `PDFToTextConverter`, and last, `PreProcessor` nodes with parameters to split documents. 

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

Then, go below to the `pipelines` section of the YAML file and create a new pipeline called `indexing`. In this pipeline, indicate how these nodes are connected to each other, Retriever, and DocumentStore. This indexing pipeline supports `.txt` and `.pdf` files and pre-processes them before loading to the Elasticsearch.

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

After completing query and indexing pipelines, add `version: ignore` to the top of the file and the pipeline YAML file is ready to run by Haystack API.

```yaml
version: ignore

components:
  - name: DocumentStore
    type: ElasticsearchDocumentStore
  - name: Retriever
    type: EmbeddingRetriever
    params:
      document_store: DocumentStore
      top_k: 5 
      embedding_model: sentence-transformers/multi-qa-mpnet-base-dot-v1
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

## Start Haystack API

Now, you can start the REST API server running the pipelines above with gunicorn server. When the startup completed, you will see it running on port 8000.


```bash
%%bash

gunicorn rest_api.application:app -b 0.0.0.0:8000 -k uvicorn.workers.UvicornWorker -t 300
```

Before continuing, test if everything is okay with the Haystack API by sending a cURL request to `/initialized` endpoint. You can use command line of your computer or tools like [Postman](https://www.postman.com/) to send cURL requests. If there is no problem, you will get a response as `true`.


```bash
%%bash

curl --request GET http://127.0.0.1:8000/initialized
```

## Index Files to Elasticsearch

Right now, the Elasticsearch instance is empty. Haystack API provides a `/file-upload` endpoint to upload files to Elasticsearch using the indexing pipeline defined in the pipeline YAML. After indexing files to Elasticsearch, you will be able to perform document search.

1. Download Example Files

Download the [example files](https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/article_txt_countries_and_capitals.zip) you will be indexing to Elasticsearch. In the folder, there are text files about countries and capitals crawled from [Wikipedia](https://en.wikipedia.org/wiki/Category:Lists_of_countries_by_continent).

2. Index Files to Elasticsearch

You can send cURL requests to `/file-upload` endpoint to upload files to Elasticsearch instance. Replace `<PATH_TO_FOLDER>` with the path to the example files on your computer and send the POST request. If the file is successfully uploaded, you will get a response as `null`.


```bash
%%bash

curl --request POST \
     --url http://127.0.0.1:8000/file-upload \
     --header 'accept: application/json' \
     --header 'content-type: multipart/form-data' \
     --form files=@<PATH_TO_FOLDER>0_Minsk.txt \
     --form meta=null
```

However, this method is not convenient to upload multiple files to Elasticsearch instance as replacing the file names in the request by hand is difficult. Instead, create a python file in the folder that you keep the example files, put the code below into the python file, and run the python script. This python code takes the name of every file in the folder and sends the POST request to the url `http://127.0.0.1:8000/file-upload`. With the `print(response.text, ":", file_name)` line, you will be able to see the names of all indexed files. Make sure you see the "Completed" text before continuing. 

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
%%bash

curl --request POST \
     --url http://127.0.0.1:8000/query \
     --header 'accept: application/json' \
     --header 'content-type: application/json' \
     --data '{
     "query": "climate in Scandinavia"
     }'
```

 As response, you will get a `QueryResponse` object consists of `query`, `answers`, and `documents`. Related [Documents](https://docs.haystack.deepset.ai/docs/documents_answers_labels#document) to your query will be under `documents` attribute of the object.

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

