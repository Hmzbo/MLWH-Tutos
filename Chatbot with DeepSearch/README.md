# Chatbot with DeepSearch functionality

## Overview

This project is an advanced conversational AI chatbot with a unique "Deep Search" capability. It leverages Google Gemini and Mistral APIs, web scraping, and document OCR to generate comprehensive, referenced reports on complex topics. The chatbot is built with Streamlit for an interactive web interface.

## Features

- **Conversational AI:** Powered by Google Gemini Flash for natural, context-aware dialogue.
- **Deep Search Tool:** Breaks down user queries, performs web searches, scrapes and cleans data, and generates detailed reports with references.
- **Downloadable Reports:** Users can download the generated Deep Search report in Markdown format.
- **Step-by-step Transparency:** The app displays a summary of each Deep Search operation, including statistics and logs.

## Installation
**Environment setup (conda):**
```
conda create -n deepsearch python=3.12
pip install -r requirements.txt
crawl4ai-setup
playwright install
```

**Environment setup (venv):**
`python -m venv .venv`

On Windows:
`.venv\Scripts\activate`
On macOS/Linux:
`source .venv/bin/activate`
```
pip install -r requirements.txt
crawl4ai-setup
playwright install
```

**Environment setup (uv):**
`uv venv .venv`

On Windows:
`.venv\Scripts\activate`
On macOS/Linux:
`source .venv/bin/activate`
```
uv pip install -r requirements.txt
crawl4ai-setup
playwright install
```

## Environment Variables

Create a `.env` file in the project root with the following keys:
```
GEMINI_API_KEY=your_google_gemini_api_key
MISTRAL_API_KEY=your_mistral_api_key
```

## Usage

1. **Start the App:**
   ```
   streamlit run app.py
   ```
2. **Interact:**  
   - Ask any question in the chat interface.
   - To trigger Deep Search, explicitly request it (e.g., "Perform a Deep Search on the impact of AI in healthcare").
   - Download the final report after Deep Search completes.

3. **Example Deep Search Prompts:**
   - "Conduct a deep search on how the retail industry has changed in the past 3 years."
   - "Perform a Deep Search on the impact of climate change on global agriculture during the last decade."
   - "Do a Deep Search on the economic effects of the COVID-19 pandemic in Europe."

## Project Structure

- `app.py` — Streamlit app entry point.
- `src/` — Core logic, tools, models, and utilities.
- `temp` — temp folder to store downloaded pdf files for OCR.
- `requirements.txt` — Python dependencies.
- `final_report.md` — Example output report.

## License

This project is licensed under the MIT License.

---

For more details, see the code and comments in [app.py](app.py) and the [src/](src/) directory.