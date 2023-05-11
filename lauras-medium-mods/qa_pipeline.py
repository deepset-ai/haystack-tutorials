from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import fetch_archive_from_http
import os
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.nodes import BM25Retriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
import pandas as pd

if __name__ == "__main__":
    # Let's first get some files that we want to query
    document_store = InMemoryDocumentStore(use_bm25=True)

    # Let's first get some files that we want to query
    doc_dir = "data/build_your_first_question_answering_system"

    # Here are some documents that we want to query with our question answering system
    fetch_archive_from_http(
        url="https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt1.zip",
        output_dir=doc_dir
    )
    
    # Transform the documents into a Haystack Document object
    files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
    indexing_pipeline = TextIndexingPipeline(document_store)
    indexing_pipeline.run_batch(file_paths=files_to_index)

    # Initialize Retriever
    retriever = BM25Retriever(document_store=document_store)

    # Initialize Reader
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
    
    # Initialize Pipeline
    pipe = ExtractiveQAPipeline(reader, retriever)

    # Generate question
    prediction = pipe.run(
                        query="Who is the father of Arya Stark?",
                        params={
                                "Retriever": {"top_k": 10},
                                "Reader": {"top_k": 5}
                            }
                        )
    
    # Build data frame
    query = [prediction['query'] for i in range(len(prediction['answers']))]
    answers_ = [prediction['answers'][i].answer for i in range(len(prediction['answers']))]
    type_ = [prediction['answers'][i].type for i in range(len(prediction['answers']))]
    score_ = [prediction['answers'][i].score for i in range(len(prediction['answers']))]
    context_ = [prediction['answers'][i].context for i in range(len(prediction['answers']))]
    offsets_in_document = [prediction['answers'][i].offsets_in_document for i in range(len(prediction['answers']))]
    offsets_in_context  =  [prediction['answers'][i].offsets_in_context for i in range(len(prediction['answers']))]



    df = pd.DataFrame(list(zip(answers_, type_, score_, context_, offsets_in_document, offsets_in_context)))

    # Rename columns
    df.columns = ['answers', 'type', 'score', 'context', 'offsets_in_document', 'offsets_in_context']

    # Save to csv
    df.to_csv('data/qa_pipeline.csv', index=False)

