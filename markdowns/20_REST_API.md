---
layout: tutorial
colab: https://colab.research.google.com/github/deepset-ai/haystack-tutorials/blob/main/tutorials/20_REST_API.ipynb
toc: True
title: "Haystack with REST API"
last_updated: 2022-11-04
level: "intermediate"
weight: 67
description: Create a production-ready pipeline and interact with REST API
category: "QA"
aliases: ['/tutorials/rest-api']
---
    

# Haystack with REST API

- **Level**: Intermediate
- **Time to complete**: 30 minutes
- **Prerequisites**: N/A
- **Nodes Used**: `ElasticsearchDocumentStore`, `EmbeddingRetriever`
- **Goal**: After completing this tutorial, you will have learned how you can interact with Haystack through REST API.

This tutorial teaches you how to create your production-ready document search `pipeline.yml` and interact with Haystack through REST API. 

First, we are going to set up the environment to run the same question answering pipeline in [Explore the World Demo](https://haystack-demo.deepset.ai/), then create a new pipeline for the new document search system.

## Setting Up the Environment

For this tutorial, we are going to need Elasticsearch and Haystack API.

### With Docker

Start up Haystack API via Docker Compose.

* **Update/install Docker and Docker Compose, then launch Docker**


```bash
%%bash

apt-get update && apt-get install docker && apt-get install docker-compose
service docker start
```

* **Clone Haystack repository**


```bash
%%bash

git clone https://github.com/deepset-ai/haystack.git
```

* **Launch Elasticsearch**

Launching Elasticsearch takes some time, so, be sure to have a `healthy` Elasticsearch container before continue. You can check the health through `docker ps` command. 

Check the other Elasticsearch initializing methods [here](https://docs.haystack.deepset.ai/docs/document_store#initialization).


```bash
%%bash

cd haystack
docker-compose elasticsearch
```

*  **Launch Haystack API**

When Elasticsearch container is ready, start the Haystack API. 


```bash
%%bash

docker-compose haystack-api
```

### Without Docker

If you prefer to not use Docker with your Haystack API, you can start the REST API server and supporting Haystack pipeline by running the gunicorn server manually with the following:


```bash
%%bash

pip install --upgrade pip
pip install 'farm-haystack[all]' ## or 'all-gpu' for the GPU-enabled dependencies
pip install -e rest_api/

brew install xpdf  ## required for `PDFToTextConverter` 

gunicorn rest_api.application:app -b 0.0.0.0:8000 -k uvicorn.workers.UvicornWorker -t 300 ## start the gunicorn server
```

You can find more information about installing Haystack at [the installation guide](https://docs.haystack.deepset.ai/docs/installation).


When setting up is done, you should notice that: 
* Haystack API: listens on port 8000
* DocumentStore (Elasticsearch): listens on port 9200

Test whether everything is okay by going to Swagger documentation for the Haystack REST API on [`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs) and trying out `/initialized` endpoint or sending a cURL request as `curl -X GET http://127.0.0.1:8000/initialized`. 

If everything is alright, you can start asking questions! Wikipedia pages about countries and capital are already indexed to Elasticsearch by the docker image we provided in [`docker-compose.yml`](https://github.com/deepset-ai/haystack/blob/main/docker-compose.yml#L22). To ask questions, you can use `/query` endpoint again via Haystack REST API UI or a cURL request.   


```bash
%%bash

curl -X 'POST' \
  'http://127.0.0.1:8000/query' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "What is the capital of Sudan",
  "params": {}
}'
```

## Create your pipeline

Now you know how REST API works, you can start customizing the pipeline

### Start Elasticsearch 

Replace the Docker image with your image or use suitable Elasticsearch image. Use `image: "docker.elastic.co/elasticsearch/elasticsearch:7.9.2"` for an empty DocumentStore.

* Go to your `docker-compose.yml` file
* Replace line 22 with `image: "docker.elastic.co/elasticsearch/elasticsearch:7.9.2"`
* Restart Elasticsearch

The new `docker-compose.yml` file should look like this:

```yaml
...
elasticsearch:
  image: "docker.elastic.co/elasticsearch/elasticsearch:7.9.2"
...
```

### New document search pipeline

We are going to create a document search pipeline. Create a new file named `document-search.haystack-pipeline.yml` in `/pipeline` folder. Do not forget to update `PIPELINE_YAML_PATH` in `config.py` with the new file name.

```python
PIPELINE_YAML_PATH = os.getenv(
    "PIPELINE_YAML_PATH", str((Path(__file__).parent / "pipeline" / "document-search.haystack-pipeline.yml").absolute())
)
```

If you are using Docker Compose for Haystack API, there are further changes you need to make in `docker-compose.yml`:
1. Create `volumes` with the path to the `document-search.haystack-pipeline.yml` file:
    ```yaml
    ...
      haystack-api:
        image: "deepset/haystack:cpu-main"
        volumes:
          - /<path_to_haystack>/haystack/rest_api/rest_api/pipeline:/home/user/rest_api/pipeline
    ...
    ``` 
2. Update the `PIPELINE_YAML_PATH` value as `/home/user/rest_api/pipeline/document-search.haystack-pipeline.yml`

This document search pipeline only requires a Retriever. Therefore, it is going to be enough for our query pipeline if we declare a DocumentStore and Retriever in `document-search.haystack-pipeline.yml`. 

Learn more about YAML files in [YAML File Definitions](https://docs.haystack.deepset.ai/docs/pipelines#yaml-file-definitions). 

```yaml
...
components:
  - name: DocumentStore
    type: ElasticsearchDocumentStore
    params:
      host: localhost
      embedding_dim: 768
  - name: Retriever
    type: EmbeddingRetriever
    params:
      document_store: DocumentStore
      embedding_model: sentence-transformers/multi-qa-mpnet-base-dot-v1 
      model_format: sentence_transformers
      top_k: 5 
...
```

And the query pipeline with a Retriever should look like this:

```yaml
...
pipelines:
  - name: query
    nodes:
      - name: Retriever
        inputs: [Query]
...
```
Be sure to have the same `name` with `QUERY_PIPELINE_NAME` variable in `config.py`.

### Indexing pipeline

You can use REST API to index your files to your document store. This requires an indexing pipeline. Add the indexing pipeline to `document-search.haystack-pipeline.yml`, then, you can use `/file-upload` endpoint to upload your files to Elasticsearch. 

Download the same demo files [here](https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/article_txt_countries_and_capitals.zip) and upload them using cURL. Check [this](https://docs.haystack.deepset.ai/docs/rest_api#indexing-documents-in-the-haystack-rest-api-document-store) documentation page for more detail about file indexing.

<aside>
⚠️ If you want to index your files directly to Elasticsearch through script, be sure to provide the same indexing pipeline with your `document-search.haystack-pipeline.yml` file for consistency between indexed files.

</aside>

The indexing pipeline should be as follows:
```yaml
...
components:
    ...
  - name: FileTypeClassifier 
    type: FileTypeClassifier
  - name: TextConverter 
    type: TextConverter
  - name: PDFConverter 
    type: PDFToTextConverter
  - name: Preprocessor 
    type: PreProcessor
    params:
      split_by: word 
      split_length: 768 
      split_overlap: 50 
      split_respect_sentence_boundary: True 

pipelines:
    ...
    - name: indexing
    nodes:
      - name: FileTypeClassifier
        inputs: [File]
      - name: TextConverter
        inputs: [FileTypeClassifier.output_1] 
      - name: PDFConverter
        inputs: [FileTypeClassifier.output_2]
      - name: Preprocessor
        inputs: [TextConverter, PDFConverter]
      - name: Retriever
        inputs: [Preprocessor]
      - name: DocumentStore
        inputs: [Retriever]
```

## Result

After merging query and the indexing pipelines, the final `document-search.haystack-pipeline.yml` file should look like this:

```yaml
version: 'ignore'

components:
  - name: DocumentStore
    type: ElasticsearchDocumentStore 
      host: localhost
      embedding_dim: 768
  - name: Retriever 
    type: EmbeddingRetriever
    params:
      document_store: DocumentStore
      embedding_model: sentence-transformers/multi-qa-mpnet-base-dot-v1
      model_format: sentence_transformers
      top_k: 5
  - name: FileTypeClassifier
    type: FileTypeClassifier
  - name: TextConverter 
    type: TextConverter
  - name: PDFConverter
    type: PDFToTextConverter
  - name: Preprocessor
    type: PreProcessor
    params:
      split_by: word 
      split_length: 768 
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
      - name: TextConverter
        inputs: [FileTypeClassifier.output_1]
      - name: PDFConverter
        inputs: [FileTypeClassifier.output_2]
      - name: Preprocessor
        inputs: [TextConverter, PDFConverter]
      - name: Retriever
        inputs: [Preprocessor]
      - name: DocumentStore
        inputs: [Retriever]
```

When you restart the Haystack API, REST API is going to load the new pipeline and the pipeline will be ready to use. 

## Voilà! Make a new query!

This query should retrieve documents about _"climate in Scandinavia"_.


```bash
%%bash

curl -X 'POST' \
  'http://localhost:8000/query' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "climate in Scandinavia",
  "params": {}
}'
```
