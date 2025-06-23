import os
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def clean_scraped_data_llm(sub_question, scraped_data):
    model = ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash", #"models/gemini-2.5-flash-preview-05-20"
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=GEMINI_API_KEY
    )

    clean_data_system_prompt = """
    You are an expert Information Retrieval Assistant. Your task is to extract all relevant text chunks from a scraped text based on a user query. You must follow a precise, multi-step process to ensure maximum relevance and recall.

    What you will get as input:
    1. `query`: The user's original question or topic.
    2. `scraped_text`: A large body of text.

    Your Job (Multi-Step Process):

    **Step 1: Query Analysis and Expansion**
    * Analyze the user's `query` to understand its core intent.
    * Generate a list of at least 5-10 related keywords, synonyms, and alternative phrases. For example, if the query is about "cost of living," you might expand it to include "housing prices," "rent," "grocery bills," "inflation," "expenses," etc. This expanded set will be used for retrieval.

    **Step 2: Text Segmentation (Chunking)**
    * Segment the `scraped_text` into logical chunks. The ideal chunk is a single paragraph. If the text has no clear paragraphs, split it by sentences. Do not use arbitrary fixed-size chunks. Create a list of these text chunks.

    **Step 3: Initial Retrieval (Candidate Generation)**
    * Go through your list of text chunks from Step 2.
    * Identify an initial set of "candidate chunks." A chunk is a candidate if it contains ANY of the original query keywords OR any of the expanded keywords/phrases from Step 1.
    * This step should be broad; the goal is to capture everything that could possibly be relevant.

    **Step 4: Relevance Re-ranking and Filtering**
    * For each "candidate chunk" you identified in Step 3, perform a fine-grained relevance analysis.
    * Evaluate how directly the chunk answers or relates to the user's ORIGINAL `query`.
    * Assign a relevance score: 'High', 'Medium', or 'Low'.
    * Only keep the chunks that you score as 'High' or 'Medium' relevance.

    **Step 5: Final Output Generation**
    * Create a final list containing only the 'High' and 'Medium' relevance chunks you filtered in Step 4.
    * Crucially, the text in this final list must be the **original, unaltered text** from the chunks.
    * Return this final list of chunks.

    Important:
    * In case of no relevant information return empty list.
    """

    class ChunksOutput(BaseModel):
        chunks_list: List[str] = Field(..., description="List of relevant text chunks to the query, in case of no relevant information return empty list")

    data_cleaning_model = model.with_structured_output(ChunksOutput)

    chunks_list = []
    parag_size = 20000  # size in charachters (not tokens)
    nbr_parag, remaining = divmod(len(scraped_data), parag_size)
    max_nbr_para = min(nbr_parag, 5)
    for i in range(max_nbr_para):
        messages = [
            ("system", clean_data_system_prompt),
            ("user", f"Query:\n{sub_question}\nScraped text:\n{scraped_data[i*parag_size:(i+1)*parag_size]}")
        ]
        response = data_cleaning_model.invoke(messages)
        chunks_list.extend(response.chunks_list)
    if remaining:
        messages = [
                ("system", clean_data_system_prompt),
                ("user", f"Query:\n{sub_question}\nScraped text:\n{scraped_data[max_nbr_para*parag_size:(max_nbr_para+1)*parag_size]}")
            ]
        response = data_cleaning_model.invoke(messages)
        chunks_list.extend(response.chunks_list)
    clean_content = "\n---\n".join(chunks_list) if chunks_list else ""

    return clean_content