from pydantic import BaseModel, Field
from langchain_core.tools import tool
import contextvars

from src.models.planner import generate_plan
from src.models.judge import judge_collected_data
from src.models.reporter import generate_sub_report, generate_final_report
from src.web_scraper import web_search, fetch_and_clean


STATUS_CONTEXT = contextvars.ContextVar('streamlit_status_context', default=None)

class DeepSearchToolInput(BaseModel):
    user_query: str = Field(description="User query to perform deep search on")

@tool("deep-search-tool", 
      description="Performs an in-depth web-based search and returns a detailed report about the user's query.", 
      args_schema=DeepSearchToolInput
      )
async def run_deepsearch(user_query: str) -> str:
    # Get the status object from the ContextVar that was set by the Streamlit app.
    status_object = STATUS_CONTEXT.get()
    deep_search_summary = {}

    # We check if status_object exists before trying to update,
    # in case the tool is called outside a Streamlit context or if contextvar wasn't set.
    if status_object:
        status_object.update(label=f"Analyzing the user query & Preparing the search plan...", state="running")
        plan = generate_plan(user_query)
        query_breakdown, sub_questions, search_queries = plan["query_breakdown"], plan["sub_questions"], plan["search_queries"]
        deep_search_summary["Step 1"] = f"**Query Breakdown**: {query_breakdown}</br> **Sub-questions**:</br>- {'</br>- '.join(sub_questions)}.</br> **Search Queries**:</br>- {'</br>- '.join(search_queries)}"

        status_object.update(label=f"Searching the web...", state="running")
        search_results = web_search(search_queries, num_results=10)
        deep_search_summary["Step 2"] = f"**Web Search Results:**</br>" + "</br>".join([f"{sq}: {urls}" for sq, urls in zip(search_queries, search_results)])

        # Data collection phase
        deep_search_summary["Step 3"] = "Collecting and cleaning data from the first url for each query:</br>"
        status_object.update(label=f"Collecting & cleaning data...", state="running")
        gathered_info = []
        for sq, urls in zip(sub_questions, search_results):
            query_data = []
            if urls:
                try:
                    url0_data = await fetch_and_clean(urls[0], sq)
                except Exception as e:
                    url0_data = "Could not fetch data from the URL."
                    print(f"Error fetching data from {urls[0]}: {e}")
                query_data.append(url0_data)
                if len(query_data[0])>50:
                    deep_search_summary["Step 3"] += f"**Sub-question:** {sq}</br>**Data from URL 1:** {query_data[0][:50]}...</br>"
                elif len(query_data[0]) == 0:
                    deep_search_summary["Step 3"] += f"**Sub-question:** {sq}</br>**Data from URL 1:** No data found.</br>"
                else:
                    deep_search_summary["Step 3"] += f"**Sub-question:** {sq}</br>**Data from URL 1:** {query_data[0]}</br>"
            gathered_info.append(query_data)


        # Data validation and supplementation
        deep_search_summary["Step 4"] = "Judging the collected data for sufficiency:</br>"
        accepted_data = []
        total_urls_scraped = len(sub_questions)
        for sq, query, data_list, urls in zip(sub_questions, search_queries, gathered_info, search_results):
            status_object.update(label=f"Judging collected data for query {query}...", state="running")
            sufficient = bool(''.join(data_list)) and judge_collected_data(data_list, sq, query)["sufficient"]
            print(f"Sub-question: {sq}\nJudgment: {sufficient}")
            deep_search_summary["Step 4"] += f"**Sub-question:** {sq}</br>- Url 1 data sufficiency **verdict**: {'Sufficient' if sufficient else 'Insufficient'}</br>"
            
            idx = 1
            while not sufficient and idx < len(urls):
                deep_search_summary["Step 4"] += f"Supplementing with data from URL {idx+1}: {urls[idx]}</br>"
                print(f"Supplementing with data from URL {idx+1}: {urls[idx]}")
                status_object.update(label=f"Supplimenting more data iteration:{idx}...", state="running")
                try:
                    data_list.append(await fetch_and_clean(urls[idx], sq))
                except Exception as e:
                    print(f"Error fetching data from {urls[idx]}: {e}")
                    print(f"Skipping URL due to error.")
                    idx += 1
                    continue  # Skip this URL if there's an error
                total_urls_scraped += 1
                if len(data_list[-1])>50:
                    deep_search_summary["Step 4"] += f"**Scraped Data from URL {idx+1}:** {data_list[-1][:50]}...</br>"
                elif len(data_list[-1]) == 0:
                    deep_search_summary["Step 4"] += f"**Scraped Data from URL {idx+1}:** No data found.</br>"
                else:
                    deep_search_summary["Step 4"] += f"**Scraped Data from URL {idx+1}:** {data_list[-1]}</br>"
                status_object.update(label=f"Verifying current data sufficiency...", state="running")
                sufficient = bool(''.join(data_list)) and judge_collected_data(data_list, sq, query)["sufficient"]
                deep_search_summary["Step 4"] += f"- Urls {[i for i in range(1, idx+2)]} data sufficiency **verdict**: {'Sufficient' if sufficient else 'Insufficient'}</br>"
                print(f"Judgment after supplementing: {sufficient}")
                idx += 1
            accepted_data.append((urls, data_list))
        print("data accepted!")
        deep_search_summary["Step 5"] = "Preparing referenced sub-reports:</br>"
        status_object.update(label=f"Preparing referenced sub-reports...", state="running")
        sub_reports = [generate_sub_report(sq, data) for sq, data in zip(sub_questions, accepted_data)]
        deep_search_summary["Step 5"] += "</br>".join([f"**Sub-question:** {sq}</br>**Sub-report:** {sub_report[:100]}..." for sq, sub_report in zip(sub_questions, sub_reports)])
        print("sub reports generated!")

        # Final report generation
        deep_search_summary["Step 6"] = "**Generating final report:**</br>"
        status_object.update(label=f"Compiling final report...", state="running")
        final_report = generate_final_report(user_query, query_breakdown, sub_questions, sub_reports)
        deep_search_summary["Step 6"] += f"**Final Report generated successfully!**"
        deep_search_summary["Statistics"] = {
            "Total Sub-questions": len(sub_questions),
            "Total URLs Searched": sum(len(urls) for urls in search_results),
            "Total URLs Scraped": total_urls_scraped,
            "Total Useful Data Chunks Collected": sum(len(data) for _, data in accepted_data),
            "Total Sub-reports Generated with length": (len(sub_reports), [len(sr) for sr in sub_reports]),
            "Final Report Length": len(final_report)
        }


    # Simulated data for testing purposes
    # final_report = "This is a simulated final report for testing purposes. If you want to use the Deep Search tool, please specify your query clearly."
    # query_breakdown = "This is a simulated query breakdown for testing purposes."
    # sub_questions = ["What is the impact of electric vehicles on urban air quality?",
    #                  "How has the adoption of electric vehicles changed over the last decade?",
    #                  "What are the economic implications of widespread electric vehicle adoption?"]
    # search_queries = ["impact of electric vehicles on urban air quality",
    #                   "adoption of electric vehicles over the last decade",
    #                   "economic implications of electric vehicle adoption"]
    
    # deep_search_summary = {
    #     "Step 1": f"**Query Breakdown**: {query_breakdown}</br> **Sub-questions**: </br>- {'</br>- '.join(sub_questions)}.</br> **Search Queries**: </br>- {'</br>- '.join(search_queries)}",
    #     "Step 2": "Simulated web search results.", 
    #     "Step 3": "Simulated data collection and cleaning.",
    #     "Step 4": "Simulated data validation and supplementation.",
    #     "Step 5": "Simulated sub-report generation.",
    #     "Step 6": "Simulated final report generation.",
    #     "Statistics": {
    #         "Total Sub-questions": 3,
    #         "Total URLs Searched": 6,
    #         "Total URLs Scraped": 4,
    #         "Total Useful Data Chunks Collected": 5,
    #         "Total Sub-reports Generated with length": (3, [150, 200, 180]),
    #         "Final Report Length": 500
    #     }
    # }

    # Note: We don't set status to 'complete' here. The 'with st.status' block in the main app
    # will handle setting the final state to 'complete' once the tool returns successfully.
    # This keeps the responsibility of managing the UI context with the UI code.
    return final_report, deep_search_summary