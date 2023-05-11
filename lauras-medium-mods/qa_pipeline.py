from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import fetch_archive_from_http
import os
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.nodes import BM25Retriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
import pandas as pd

def generate_dataframe_stack(question, num_possible_answers=1): 
    """
    Generates a dataframe with the answers to a query

    Parameters
    ----------
    query : str
        The question to be answered

    Returns
    -------
    df : pandas dataframe
        A dataframe with the answers to the query
    """

    # Generate question
    prediction = pipe.run(
                        query=question,
                        params={
                                "Retriever": {"top_k": 10},
                                "Reader": {"top_k": num_possible_answers}
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

    # Create dataframe
    df = pd.DataFrame(list(zip(query, answers_, type_, score_, context_, offsets_in_document, offsets_in_context)))

    # Rename columns
    df.columns = ['query','answers', 'type', 'score', 'context', 'offsets_in_document', 'offsets_in_context']

    return df

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

    # Set up a few questions
    questions = [
        "What are the sigils and mottos of the major houses in Game of Thrones?",
        "Who was Jon Snow's mother?",
        "Who pushed Bran Stark out of the window in the first season, and why?",
        'Which character said the famous line, "When you play the game of thrones, you win or you die"?',
        "Who was the Mad King, and what was his real name?",
        "How many dragons did Daenerys Targaryen have, and what were their names?",
        "What is the name of the massive ice wall that protects the Seven Kingdoms from the creatures of the North?",
        "Who was the leader of the White Walkers?",
        "Who is the youngest Stark child?",
        "Which characters are members of the Night's Watch at the start of the series?",
        "What are the faceless men, and which character becomes associated with them?",
        "What is the name of Arya's direwolf?",
        "What is the ancestral sword of House Stark?",
        "Who resurrects Jon Snow after he is murdered?",
        "What is the real name of Littlefinger?",
        "How does Tyrion Lannister kill his father?",
        "Who kills the Night King, and with what weapon?",
        "Which character ends up ruling Westeros at the end of the series?",
        "Who were the three members of Robert's Rebellion?",
        "What are the names of the two cities Daenerys liberates before reaching Westeros?"
    ]

    # Generate dataframe
    question_df_list = list()
    for question in questions:
        df = generate_dataframe_stack(question, num_possible_answers=1)
        question_df_list.append(df)

    # Concatenate dataframes
    answers_df = pd.concat(question_df_list)

    print(answers_df)

    # Save to csv
    answers_df.to_csv('data/tabular/qa_pipeline.csv', index=False)

