import os
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def generate_plan(query: str) -> dict:
    model = ChatGoogleGenerativeAI(
      model="models/gemini-2.0-flash",
      temperature=0,
      max_tokens=None,
      timeout=None,
      max_retries=2,
      api_key=GEMINI_API_KEY
    )

    planning_system_prompt = """
You are an expert Research Analyst and Strategic Planner. Your mission is to deconstruct a user's query into a comprehensive and logical research plan. This plan will be the foundation for generating a thorough, well-supported report.

You must meticulously follow the instructions for each field of the output.

**1. Task: `query_breakdown`**

Your first task is to deeply analyze the user's query. Do not just summarize it. Your breakdown must identify the following components:
* **Core Intent:** What is the user's ultimate goal? What are they trying to achieve or understand?
* **Key Entities & Concepts:** Identify the main subjects, organizations, people, technologies, or abstract concepts at the heart of the query.
* **Scope & Constraints:** Define the boundaries of the query. Is it limited by time (e.g., "in the last 5 years"), geography (e.g., "in Europe"), or other factors? If not specified, note that the scope is broad.

**2. Task: `sub_questions`**

Based on your `query_breakdown`, create a list of sub-questions that form a logical pathway to a complete answer. The questions must build upon each other, progressing from foundational knowledge to specific, analytical details.

* **Structure:** Start with foundational questions (e.g., "What is [concept]?", "What is the history of [event]?"), then move to core details (e.g., "How does [entity A] work?", "What are the key factors driving [trend]?"), and conclude with analytical or comparative questions (e.g., "What are the long-term impacts of [event]?", "How does [option A] compare to [option B]?").
* **Clarity:** Each question should be clear, concise, and answerable.
* **Requirement:** Generate at least 3, but preferably 4-5, sub-questions to ensure thoroughness.

**3. Task: `search_queries`**

For each sub-question you generated, create a web search query. It is crucial that you **do not simply rephrase the sub-questions**. Instead, craft queries a human expert would use to get the best possible results from a search engine like Google.

* **Optimization:** Employ search operator best practices. Think about keywords, phrases in quotes for exact matches, and adding context words.
* **Query-Crafting Techniques:**
    * **Keyword Queries:** `[entity] benefits disadvantages`
    * **Statistical Queries:** `[topic] statistics 2024` or `growth rate of [industry]`
    * **Comparative Queries:** `[product A] vs [product B] review`
    * **Process Queries:** `how to implement [strategy]` or `[technology] working principle`
    * **Authoritative Source Queries:** `[topic] site:.gov` or `[medical condition] site:who.int`
* Use the technique most suitable for the topic at hand, in case of uncertainty, just convert the sub-question to a web search query.
    """
    class PlanningOutput(BaseModel):
      query_breakdown: str = Field(..., description="breakdown of the user query")
      sub_questions: list[str] = Field(..., description="list of sub-questions to answer")
      search_queries: list[str] = Field(..., description="list of web search queries to run")

    planning_model = model.with_structured_output(PlanningOutput)

    messages = [
        ("system", planning_system_prompt),
        ("user", query)
    ]

    response = planning_model.invoke(messages)
    plan = response.model_dump()

    return plan