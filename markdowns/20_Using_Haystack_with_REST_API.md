---
layout: tutorial
featured: True
colab: False
toc: True
title: "Using Haystack with REST API"
last_updated: 2023-01-11
level: "advanced"
weight: 115
description: Create a production-ready pipeline and interact with Haystack REST API.
category: "QA"
aliases: ['/tutorials/using-haystack-with-rest-api']
download: "/downloads/20_Using_Haystack_with_REST_API.ipynb"
completion_time: 30 min
---
    


- **Level**: Advanced
- **Time to complete**: 30 minutes
- **Prerequisites**: Basic understanding of Docker and basic knowledge of Haystack pipelines. 
- **Nodes Used**: `ElasticsearchDocumentStore`, `EmbeddingRetriever`
- **Goal**: After you complete this tutorial, you will have learned how to interact with Haystack through REST API.

## Overview

Learn how you can interact with Haystack through REST API. This tutorial introduces you to all the concepts needed to build an end-to-end document search application.  

With Haystack, you can apply the latest NLP technology to your own data and create production-ready applications. Building an end-to-end NLP application requires the combination of multiple concepts:
* **DocumentStore** is the component in Haystack responsible for loading and storing text data in the form of [Documents](https://docs.haystack.deepset.ai/docs/documents_answers_labels#document). In this tutorial, the DocumentStore uses Elasticsearch behind the scene.
* **Haystack pipelines** convert files into Documents, index them to the DocumentStore, and run NLP tasks such as question answering and document search.
* **REST API**, as a concept, makes it possible for applications to interact with each other by handling their queries and returning responses. There is `rest_api` application within Haystack that exposes Haystack's functionalities through a RESTful API.
* **Docker** simplifies the environment setup needed to run Elasticsearch and Haystack API.



## Preparing the Environment

1. Install [Docker Compose](https://docs.docker.com/compose/) and launch Docker.
If you installed Docker Desktop, just start the application. Run `docker info` to see if Docker is up and running:

   ```bash
   docker info
   ```

2. Download the *docker-compose.yml* file. Haystack provides a *docker-compose.yml* file that defines services for Haystack API and Elasticsearch. 
    1. Create a new folder called *doc-search* in a directory where you want to keep all tutorial related files.
    2. Save the latest [*docker-compose.yml*](https://github.com/deepset-ai/haystack/blob/main/docker-compose.yml) file from GitHub into the folder. To save the *docker-compose.yml* file into the directory directly, run:

         ```bash
         curl --output docker-compose.yml https://raw.githubusercontent.com/deepset-ai/haystack/main/docker-compose.yml
         ```

    Here's what the */doc-search* folder should look like:
    ```
    /doc-search
    └── docker-compose.yml
    ```

Now that your environment's ready, you can start creating your indexing and query pipelines.

## Creating the Pipeline YAML File

You can define components and pipelines using YAML code that Haystack translates into Python objects. In a pipeline YAML file, the `components` section lists all pipeline nodes and the `pipelines` section defines how these nodes are connected to each other. Let's start with defining two different pipelines, one to index your documents and another one to query them. We'll use one YAML file to define both pipelines.

1. Create a document search pipeline. This will be your query pipeline:
   1. In the newly created *doc-search* folder, create a file named *document-search.haystack-pipeline.yml*. The *docker-compose.yml* file and the new pipeline YAML file should be on the same level in the directory:

      ```
      /doc-search
      ├── docker-compose.yml
      └── document-search.haystack-pipeline.yml
      ```

   2. Provide the path to *document-search.haystack-pipeline.yml* as the `volume` source value in the *docker-compose.yml* file. The path must be relative to *docker-compose.yml*. As both files are in the same directory, the source value will be `./`. 

      ```yaml
      haystack-api:
        ...
        volumes:
          - ./:/opt/pipelines
      ```

   3. Update the `PIPELINE_YAML_PATH` variable in *docker-compose.yml* with the name of the pipeline YAML file. The `PIPELINE_YAML_PATH` variable tells `rest_api` which YAML file to run. 

      ```yaml
      environment:
        ...
        - PIPELINE_YAML_PATH=/opt/pipelines/document-search.haystack-pipeline.yml
        ...
      ```
   4. Define the pipeline nodes in the `components` section of the file. A document search pipeline requires a DocumentStore and a Retriever. Our pipeline will use `ElasticsearchDocumentStore` and `EmbeddingRetriever`:

      ```yaml
      components:
        - name: DocumentStore # How you want to call this node here
          type: ElasticsearchDocumentStore # This is the Haystack node class
          params: # The node parameters
            embedding_dim: 384 # This parameter is required for the embedding_model
        - name: Retriever
          type: EmbeddingRetriever
          params:
            document_store: DocumentStore
            top_k: 10
            embedding_model: sentence-transformers/all-MiniLM-L6-v2
      ```

   5. Create a query pipeline in the `pipelines` section. Here, `name` refers to the name of the pipeline, and `nodes` defines the order of the nodes in the pipeline: 

      ```yaml
      pipelines:
        - name: query 
          nodes:
            - name: Retriever
              inputs: [Query]
      ```

2. In the same YAML file, create an indexing pipeline. This pipeline will index your documents to Elasticsearch through `rest_api`. 
   1. Define `FileTypeClassifier`, `TextConverter`, and `PreProcessor` nodes for the pipeline:

      ```yaml
      components:
        ...
        - name: FileTypeClassifier
          type: FileTypeClassifier
        - name: TextFileConverter
          type: TextConverter
        - name: Preprocessor
          type: PreProcessor
          params: # These parameters define how you want to split your documents
            split_by: word
            split_length: 250
            split_overlap: 30 
            split_respect_sentence_boundary: True 
      ```

   2. In the `pipelines` section of the YAML file, create a new pipeline called `indexing`. In this pipeline, indicate how the nodes you just defined are connected to each other, Retriever, and DocumentStore. This indexing pipeline supports *.TXT* files and preprocesses them before loading to Elasticsearch.

      ```yaml
      pipelines:
        ...
        - name: indexing
          nodes:
            - name: FileTypeClassifier
              inputs: [File]
            - name: TextFileConverter
              inputs: [FileTypeClassifier.output_1]
            - name: Preprocessor
              inputs: [TextFileConverter]
            - name: Retriever
              inputs: [Preprocessor]
            - name: DocumentStore
              inputs: [Retriever]
      ```

3. After creating query and indexing pipelines, add `version: 1.12.1` to the top of the file. This is the Haystack version that comes with the Docker image in the *docker-compose.yml*. Now, the pipeline YAML is ready.

```yaml
version: 1.12.1

components:
  - name: DocumentStore
    type: ElasticsearchDocumentStore
    params:
      embedding_dim: 384
  - name: Retriever
    type: EmbeddingRetriever
    params:
      document_store: DocumentStore
      top_k: 10 
      embedding_model: sentence-transformers/all-MiniLM-L6-v2
  - name: FileTypeClassifier
    type: FileTypeClassifier
  - name: TextFileConverter
    type: TextConverter
  - name: Preprocessor
    type: PreProcessor
    params:
      split_by: word
      split_length: 250
      split_overlap: 30 
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
      - name: Preprocessor
        inputs: [TextFileConverter]
      - name: Retriever
        inputs: [Preprocessor]
      - name: DocumentStore
        inputs: [Retriever]
```

Feel free to play with the pipeline setup later on. Add or remove some nodes, change the parameters, or add new ones. For more options for nodes and parameters, check out [Haystack API Reference](https://docs.haystack.deepset.ai/reference/answer-generator-api).

## Launching Haystack API and Elasticsearch

Pipelines are ready. Now it's time to start Elasticsearch and Haystack API.
1. Run `docker-compose up` to start the `elasticsearch` and `haystack-api` containers. This command installs all the necessary packages, sets up the environment, and launches both Elasticsearch and Haystack API. Launching may take 2-3 minutes. 

   ```bash
   docker-compose up
   ```

2. Test if everything is OK with the Haystack API by sending a cURL request to the `/initialized` endpoint. If everything works fine, you will get `true` as a response.

   ```bash
   curl --request GET http://127.0.0.1:8000/initialized
   ```


Both containers are initialized. Time to fill your DocumentStore with files.  

## Indexing Files to Elasticsearch

Right now, your Elasticsearch instance is empty. Haystack API provides a `/file-upload` endpoint to upload files to Elasticsearch. This endpoint uses the indexing pipeline you defined in the pipeline YAML. After indexing files to Elasticsearch, you can perform document search.

1. Download the [example files](https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/article_txt_countries_and_capitals.zip) to the *doc-search* folder. The .zip file contains text files about countries and capitals crawled from [Wikipedia](https://en.wikipedia.org/wiki/Category:Lists_of_countries_by_continent).

   ```
   /doc-search
   ├── docker-compose.yml
   ├── document-search.haystack-pipeline.yml
   └── /article_txt_countries_and_capitals
       ├── 0_Minsk.txt
       └── ...
   ```

2. Index files to Elasticsearch. You can send cURL requests to the `/file-upload` endpoint to upload files to the Elasticsearch instance. If the file is successfully uploaded, you will get `null` as a response.

   ```bash
   curl --request POST \
        --url http://127.0.0.1:8000/file-upload \
        --header 'accept: application/json' \
        --header 'content-type: multipart/form-data' \
        --form files=@article_txt_countries_and_capitals/0_Minsk.txt \
        --form meta=null
   ```

   This method is not the best one if you have multiple files to upload. That's because you need to replace file names in the request by hand. Instead, you can run a command that takes all *.TXT* files in the *article_txt_countries_and_capitals* folder and sends a POST request to index each file:   

   ```bash
   find ./article_txt_countries_and_capitals -name '*.txt' -exec \
        curl --request POST \
             --url http://127.0.0.1:8000/file-upload \
             --header 'accept: application/json' \
             --header 'content-type: multipart/form-data' \
             --form files="@{}" \
             --form meta=null \;
   ```

## Querying Your Pipeline

That's it, the application is ready! Send another POST request to retrieve documents about _"climate in Scandinavia"_: 

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

```python
{
  "query": "climate in Scandinavia",
  "answers": [],
  "documents": [
    {
      "id": "24904f783ea4b90a47c33434a3e9df7a",
      "content": "Because of Sweden's high latitude, the length of daylight varies greatly. North of the Arctic Circle, the sun never sets for part of each summer, and it never rises for part of each winter. In the capital, Stockholm, daylight lasts for more than 18 hours in late June but only around 6 hours in late December. Sweden receives between 1,100 and 1,900 hours of sunshine annually...",
      "content_type": "text",
      "meta": {
        "_split_id": 33,
        "name": "43_Sweden.txt"
      },
      "score": 0.5017639926813274
    },
    ...
  ]
}
```  

Congratulations! You have created a proper search system that runs using Haystack REST API.

## About us

This [Haystack](https://github.com/deepset-ai/haystack/) tutorial was made with love by [deepset](https://deepset.ai/) in Berlin, Germany

We bring NLP to the industry via open source!  
Our focus: Industry specific language models & large scale QA systems.  
  
Some of our other work: 
- [German BERT](https://deepset.ai/german-bert)
- [GermanQuAD and GermanDPR](https://deepset.ai/germanquad)
- [FARM](https://github.com/deepset-ai/FARM)

Get in touch:
[Twitter](https://twitter.com/deepset_ai) | [LinkedIn](https://www.linkedin.com/company/deepset-ai/) | [Discord](https://haystack.deepset.ai/community) | [GitHub Discussions](https://github.com/deepset-ai/haystack/discussions) | [Website](https://deepset.ai)

By the way: [we're hiring!](https://www.deepset.ai/jobs)

