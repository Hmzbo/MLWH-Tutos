import os
import sys
import asyncio
import time
import random
import markdown
import numpy as np
import streamlit as st
from streamlit_float import *
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.tools import run_deepsearch, STATUS_CONTEXT

st.set_page_config(layout="wide")

# Load environment variables from .env file
load_dotenv()
# initialize float feature/capability
float_init()


# Configuration: GEMINI_API_KEY should be set in environment or Streamlit secrets
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if sys.platform == "win32":
    # Set the event loop policy for Windows to ensure subprocess support
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


# Initialize Gemini model (cached to prevent re-initialization)
@st.cache_resource
def load_model():
    model = ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=GEMINI_API_KEY
    )
    return model.bind_tools([run_deepsearch])

# System prompt
chatbot_system_prompt = """
You are a helpful conversational assistant. Your role is to assist the user with their questions and follow their instructions as clearly and accurately as possible.

You have access to a special tool called "deep-search-tool", which can perform an in-depth investigation on a topic. Only use this tool when the user explicitly requests a "Deep Search" or using the trigger words "DeepSearch" or "Deep Search" in their query.

If the user asks how Deep Search works, respond by saying:  
"I have access to a special tool that performs the Deep Search, but I don't know exactly how it works under the hood."

Avoid assuming the user wants a Deep Search unless they clearly say so.

Important:
- Once you decided to use the deep search tool, ask the user one last question to make sure they are using the tool correctly.
Say for example:
Are you sure you want me to perform Deep Search on: (here put the query you will pass to the tool)?
If the answer is yes, then proceed with tool calling. If the answer is no, then don't use the tool.
- If the user asks about deep search then don't assume that they wants to use it. Example:
user: Can you perform deep search?
-> don't assume that they wants to use it. Just respond to the question: Yes, I cant perfrom Deep Search Operation.
or:
user: I want you to perform deep search.
-> the user didn't specify what to search. So don't trigger the tool yet, just tell them you can perform deep search but you need a query or a topic.
"""

# Create prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", chatbot_system_prompt),
    ("placeholder", "{history}"),
    ("human", "{input}")
])

# Initialize chain lazily
@st.cache_resource
def init_chain():
    model = load_model()
    chain = prompt_template | model
    return chain

async def call_deepsearch_tool(tool_args):
    return await run_deepsearch.ainvoke(tool_args)


# Tools dict
tools_dict = {"deep-search-tool": call_deepsearch_tool}


# Streamed response emulator
def response_generator(response):
    for word in response.split(" "):
        sleep_time = random.uniform(0.005, 0.03)  # Random sleep time between 10ms and 50ms
        yield word + " "
        time.sleep(sleep_time)

# Initialize chat history
if "display_history" not in st.session_state:
    st.session_state.display_history = []
if "internal_history" not in st.session_state:
    st.session_state.internal_history = []

# This class is just for testing purposes, to simulate the response from the model.
# class Response():
#     def __init__(self, content):
#         self.content = content
#         self.tool_calls = [{"name": "deep-search-tool", "args": {"user_query": "What is the impact of climate change on global agriculture?"}, "id": "12345"}]

col1, col2, col3 = st.columns([1.2, 1.5, 1], vertical_alignment="top")
with col2:
    # Set title
    st.title("🤖 Chatbot with Deep Search 🤖")
st.divider()

colI, colII, colIII = st.columns([0.2, 1.5, 1.5])
with colII:
    st.header("Chat with AI")
    st.caption("Powered by Google Gemini Flash :zap:")
    # Display chat messages from history on app rerun
    for message in st.session_state.display_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    # Accept user input
    input_container = st.container()
    with input_container:
        coli, colii, coliii = st.columns([0.6, 1, 0.8])
        with colii:
            report_exist = 'final_report' in st.session_state
            st.download_button(
                label="Download Deep Search Report",
                help="Ask the AI to perdorm Deep Search, then you can download the generated report.",
                data=st.session_state.get("final_report", ""),
                file_name="final_report.md",
                on_click="ignore",
                type="primary",
                icon=":material/download:",
                use_container_width=True,
                disabled=not report_exist  # Disable if final_report is not set
            )
            user_input = st.chat_input("How can I assist you today?")
    input_container.float("bottom: 0; padding: 20px; transform: translate(-25%, 0%);")
    
    if user_input:
        # Add user message to chat history
        st.session_state.display_history.append({"role": "user", "content": user_input})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.write(user_input)

        chain = init_chain()
        try:
            # Invoke the chain with current input and history
            response = chain.invoke({
                "history": st.session_state.internal_history,
                "input": user_input
            })

            # Simulated response for testing purposes
            # response = Response(content="This is a simulated response for testing purposes. If you want to use the Deep Search tool, please specify your query clearly.")

            # Check if the response contains tool calls
            tool_calls = getattr(response, "tool_calls", None)
            if tool_calls:
                selected_tool = tools_dict[tool_calls[0]["name"].lower()]
                with st.status("Calling Deep Search Tool...", expanded=True) as status:
                    token = STATUS_CONTEXT.set(status)
                    try:
                        generated_report, deep_search_summary = asyncio.run(selected_tool(tool_calls[0]["args"]))
                        status.update(label="Deep Search complete!", state="complete", expanded=False)
                    except Exception as e:
                        status.update(label=f"Deep Search failed: {str(e)}", state="error", expanded=True)
                        raise
                    finally:
                        # Crucially, reset the ContextVar to its previous value (or default)
                        # This prevents cross-contamination if other async tasks or reruns occur.
                        STATUS_CONTEXT.reset(token)
                # Update history with both user input, AI response, and Tool response
                st.session_state.internal_history.append(HumanMessage(content=user_input))
                st.session_state.internal_history.append(AIMessage(content=response.content))
                st.session_state.internal_history.append(ToolMessage(content=generated_report, tool_call_id=tool_calls[0]["id"]))
                st.session_state.internal_history.append(AIMessage(content=generated_report))
                st.session_state.display_history.append({"role": "ai", "content": generated_report})
                st.session_state.final_report = generated_report  # Store final report for download
                st.session_state.summary = deep_search_summary  # Store summary for display
                # Display user message in chat message container
                with st.chat_message("ai"):
                    st.write_stream(response_generator(generated_report))
                    st.rerun()
                    
            else:
                # Update history with both user input and AI response
                st.session_state.internal_history.append(HumanMessage(content=user_input))
                st.session_state.internal_history.append(AIMessage(content=response.content))
                st.session_state.display_history.append({"role": "ai", "content": response.content})
                # Display user message in chat message container
                with st.chat_message("ai"):
                    st.write_stream(response_generator(response.content))
        except Exception as e:
            error_message = f"Sorry, an error occurred: {str(e)}"
            st.error(error_message)

with colIII:
    if 'final_report' in st.session_state:
        summary_container = st.container()
        with summary_container:
            summary_dict = st.session_state.get("summary", "")
            summary_md1 = f"""
## :red[Deep Search Summary]
This section provides a summary of the Deep Search operation performed by the AI.
### :red[Statistics]:
"""
            st.markdown(summary_md1, unsafe_allow_html=True)
            a, b, c = st.columns(3)
            d, e, f = st.columns(3)
            a.metric("Total Sub-questions", f"{summary_dict["Statistics"]["Total Sub-questions"]}", "")
            b.metric("Total URLs Searched", f"{summary_dict["Statistics"]["Total URLs Searched"]}", "")
            c.metric("Total URLs Scraped", f"{summary_dict["Statistics"]["Total URLs Scraped"]}", "")
            d.metric("Total Useful Data Chunks Collected", f"{summary_dict["Statistics"]["Total Useful Data Chunks Collected"]}", "")
            e.metric("Sub-report Avg. Length", f"{int(np.mean(summary_dict["Statistics"]["Total Sub-reports Generated with length"][1]))}", "")
            f.metric("Final Report Length", f"{summary_dict["Statistics"]["Final Report Length"]}", "")

            summary_md2 = f"""
### :red[Steps Logs]:
- **:blue[Step 1]:**</br>{markdown.markdown(summary_dict["Step 1"])}
- **:blue[Step 2]:**</br>{markdown.markdown(summary_dict["Step 2"])}
- **:blue[Step 3]:**</br>{markdown.markdown(summary_dict["Step 3"])}
- **:blue[Step 4]:**</br>{markdown.markdown(summary_dict["Step 4"])}
- **:blue[Step 5]:**</br>{markdown.markdown(summary_dict["Step 5"])}
- **:blue[Step 6]:**</br>{markdown.markdown(summary_dict["Step 6"])}
            """
            st.markdown(summary_md2, unsafe_allow_html=True)

    readme = """
## :primary[What is Deep Search?]
Deep Search is a powerful tool designed to perform in-depth investigations on complex topics. 
It breaks down user queries into manageable sub-questions, conducts web searches, and collects relevant data to generate comprehensive reports.
## :primary[How to use Deep Search?]
To use the Deep Search functionality, simply ask the AI to perform a "Deep Search" on a specific topic or question. For example, you can say:
- "Conduct a deep search on how the retail industry has changed in the past 3 years."
- "Perform a Deep Search on the impact of climate change on global agriculture during the last decade."
- "Do a Deep Search on the economic effects of the COVID-19 pandemic in Europe."

:primary[**Notes:**]
- The Deep Search tool usually takes a few minutes to complete, depending on the complexity of the query and the amount of data collected.
- Deep Search is only triggered when the user explicitly requests it.
    """
    if 'final_report' not in st.session_state:
        st.markdown(readme)

    
