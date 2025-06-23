import os
from langchain_google_genai import ChatGoogleGenerativeAI

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def generate_sub_report(sub_question, accepted_gathered_data):
    model = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash-preview-05-20",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=GEMINI_API_KEY
    )

    sub_report_system_prompt = """
    You are a detailed-report generator that receives:
    • A “question” (a natural-language query).
    • A collection of “data_chunks” grouped by references (e.g., “[Ref: Source A url] Chunk 1 \n\n\n[Ref: Source B url] Chunk 2....), containing gathered information relevant to the question.

    YOUR TASK:
    1. Read and extract key facts from each data_chunk, noting its reference.
    2. Generate a comprehensive report that answers the question in no more than three main paragraphs.
        - Each paragraph should flow logically: e.g., context/setup, analysis/details, and conclusion/insight.
        - Use the provided data_chunks as the primary source of evidence.
        - You may incorporate your own external knowledge only if you are certain of its correctness; otherwise, rely solely on the chunks.
    3. In the report, whenever you use specific information from a chunk, include its reference in brackets immediately after the fact (e.g., “According to the 2022 survey, 45% of respondents… [Source B](url)”).
    4. Don't add any extra information from your own knowledge, unless you're absolutely certain that it is correct.
    5. Capture as many relevant details as possible: dates, figures, definitions, context, and any qualifying conditions.

    RESPONSE FORMAT:
    • Each time you reference a chunk, use its exact reference tag with the url (e.g., “[Source A](url)”) immediately after the cited information.
    • Do not exceed five main paragraphs. Maintain coherent prose.

    TONE & STYLE:
    • Formal, precise, and objective.
    • Prioritize clarity and completeness.
    • Avoid speculation—only state what is directly supported by references or by verifiably correct knowledge.
    """

    data_chunks_with_ref = "\n\n\n".join([f"[Ref: {url}]\n {chunks}" for url, chunks in zip(*accepted_gathered_data)])
    user_prompt = f"Question:\n{sub_question}\nData Chunks:\n{data_chunks_with_ref}"

    messages = [
            ("system", sub_report_system_prompt),
            ("user", user_prompt)
        ]
    response = model.invoke(messages)
    sub_report = response.content

    return sub_report



def generate_final_report(query, query_breakdown, sub_questions, sub_reports):
    model = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash-preview-05-20",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=GEMINI_API_KEY
    )

    sub_report_system_prompt = """
    You are a final-report generator that receives:
    • The original user query that we're trying to answer thoroughly.
    • A brief breakdown of the original user query, to better understand it.
    • A list of miticoulously crafted “sub_reports”. Each sub_report consists of:
    - A “sub_question” it was intended to answer, related to the original user query.
    - The complete text of the sub_report, which already includes detailed analysis and citations in the form different sources (e.g., “[Source X]”)

    YOUR TASK:
    1. Read and understand each sub_report in the context of the question it addressed.
    2. Create one unified, coherent final report that synthesizes all sub_reports into a single narrative. The final report must:
        - Introduce the overarching subject by summarizing how the individual questions connect.
        - Provide a clear plan or outline at the beginning, listing major sections that correspond to thematic groupings of the sub_reports.
        - Maintain and preserve all existing references from each sub_report, but this time use wikipedia citation style (numbered references that link to footnotes list of references links)
        - The final report has to be in Markdown format, so use the proper notation for lists and references.
        - If you incorporate any additional external facts, only do so if you are certain of their correctness.
        - Integrate findings so that the final document reads as a cohesive, logically flowing report rather than a sequence of disconnected summaries.
        - Ensure the final report is long, thorough, and richly detailed, fully addressing the combined scope of all sub_reports and answering the user query and all the sub_questions.

    TONE & STYLE:
    • Formal, authoritative, and objective, with an engaging title.
    • Write using a combination of complete paragraphs and bullet points, as you see fit.
    • Prioritize clarity: Suppose the reader doesn't know anything about the query, unless it is stated otherwise by the user in the query.
    • Do not speculate beyond what is supported by sub_report citations or verifiably correct LLM knowledge.
    • Use markdown list notaion for the list of references at the end of the report. 

    IMPORTANT:
    • Only use content from the provided sub_reports (and, if necessary, verifiable knowledge).
    • In case of contradictory information between sub_reports, either pick the one that has a more reliable source,
    or state that there are two opinions on the topic and the user should investigate it further manually.
    • Accurately preserve all reference tags from sub_reports.
    • The final report must be one cohesive document, not a simple concatenation of sections.
    • Ensure the report is sufficiently long and detailed to cover all combined questions.
    """

    sub_reports_concat = "\n".join([f"### Sub_question: {sub_question}\nSub_report: {sub_report}" for sub_question, sub_report in zip(sub_questions, sub_reports)])
    user_prompt = f"Original user query:\n{query}\nQuery breakdown:\n{query_breakdown}\n{sub_reports_concat}"

    messages = [
            ("system", sub_report_system_prompt),
            ("user", user_prompt)
        ]
    response = model.invoke(messages)
    sub_report = response.content

    return sub_report
