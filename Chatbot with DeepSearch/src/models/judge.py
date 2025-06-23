import os
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def judge_collected_data(gathered_info, sub_question, query):
    model = ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=GEMINI_API_KEY
    )
    judgement_system_prompt = """
    You are a judgment-oriented assistant whose sole role is to evaluate whether provided data chunks are sufficient to answer a given question thoroughly. Follow these rules:

    1. INPUT FORMAT
    • The user will supply:
        - A “question” (a natural-language query).
        - A “web search query” (query used to perform web search to collect data)
        - A list of “data_chunks”, where each chunk is a fragment of collected information. These chunks are not guaranteed to be coherent narrative—treat them as separate pieces of evidence.
    • Example input (not part of your prompt):
        Question:
        What are the demographic trends of electric vehicle adoption in urban areas over the last five years?
        Web search query:
        electric vehicle adoption in urban areas
        Data Chunks:
        Data from City A's transportation survey (2020): 12% EV penetration ...
        ---
        Academic paper excerpt: In 2019, urban EV buyers skewed younger ...
        ---
        News article (2024) about tax incentives affecting EV sales ...

    2. PROCESS
    • Treat the “data_chunks” as discrete units; do NOT assume they connect seamlessly.
    • For each chunk, extract the key facts, dates, and context.
    • Evaluate coverage: timeline span, geographic scope, demographic variables, methodologies, sample sizes, and relevance to the question.
    • Do not attempt to “answer” the question yourself using outside knowledge; only judge sufficiency of the provided chunks.

    3. Output
    - State whether the data is sufficient (True or False)
    """
    class JudgementOutput(BaseModel):
        sufficient: bool = Field(..., description="Whether the collected data is sufficient or not")

    judgement_model = model.with_structured_output(JudgementOutput)
    gethered_info_concat = '---\n'.join(gathered_info)
    messages = [
        ("system", judgement_system_prompt),
        ("user", f"Question:\n{sub_question}\nWeb search query:\n{query}\nData Chunks:\n{gethered_info_concat}")
    ]

    response = judgement_model.invoke(messages)
    judgment = response.model_dump()

    return judgment