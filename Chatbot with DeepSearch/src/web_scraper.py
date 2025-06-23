import requests
import json
from typing import Tuple
from googlesearch import search
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, LLMConfig
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from src.utils import pdf_ocr
from src.models.cleaner import clean_scraped_data_llm


def should_skip_url(url: str) -> Tuple[bool, str]:
    # New check: Skip non-HTTPS URLs
    if not url.lower().startswith('https://'):
        print(f"[INFO] Skipping URL (non-https): {url}")
        return True, 'non-https'

    # Existing logic below (unchanged)
    NON_TEXTUAL_MIME_TYPES = [
        'application/zip',
        'application/x-rar-compressed',
        'application/octet-stream',
        'image/',
        'video/',
        'audio/',
        'font/',
    ]

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        head_response = requests.head(url, allow_redirects=True, timeout=5, headers=headers)
        head_response.raise_for_status()
        content_type = head_response.headers.get('Content-Type', '').lower()

        for mime_type in NON_TEXTUAL_MIME_TYPES:
            if content_type.startswith(mime_type):
                detected_type = mime_type.strip('/')
                print(f"[INFO] Skipping URL (non-textual content type: {detected_type}): {url}")
                return True, detected_type

        with requests.get(url, allow_redirects=True, timeout=10, headers=headers, stream=True) as get_response:
            get_response.raise_for_status()
            final_content_type = get_response.headers.get('Content-Type', '').lower()

            for mime_type in NON_TEXTUAL_MIME_TYPES:
                if final_content_type.startswith(mime_type):
                    detected_type = mime_type.strip('/')
                    print(f"[INFO] Skipping URL (non-textual content type on GET: {detected_type}): {url}")
                    return True, detected_type

            if 'application/pdf' in final_content_type:
                print(f"[SUCCESS] URL content is valid (pdf): {url}")
                return False, 'pdf'
            else:
                print(f"[SUCCESS] URL content is valid (html/text): {url}")
                return False, 'html'

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Skipping URL due to request error: {url}\n\tDetails: {e}")
        return True, 'error'
    except Exception as e:
        print(f"[ERROR] Skipping URL due to an unexpected error: {url}\n\tDetails: {e}")
        return True, 'error'

def web_search(search_queries: list[str], num_results: int = 5) -> list[str]:
    """
    Return a list of top N URLs from a Google search.
    """
    results = []
    for search_query in search_queries:
      query_related_urls = []
      urls = search(search_query, num_results=num_results, timeout=5, unique=True)
      for url in urls:
        verdict, url_type = should_skip_url(url)
        if not verdict:
          query_related_urls.append((url_type, url))
      results.append(query_related_urls)
    return results


async def crawl4ai_func(url):
    # Define markdown generator
    md_generator = DefaultMarkdownGenerator(
        options={
            "ignore_links": True,
            "skip_internal_links": True,
            "ignore_images": True
        }
    )

    # Build the crawler config
    crawl_config = CrawlerRunConfig(
        markdown_generator=md_generator,
        scraping_strategy=LXMLWebScrapingStrategy(),
        cache_mode=CacheMode.BYPASS
    )

    # Create a browser config if needed
    browser_cfg = BrowserConfig(headless=True)

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(
            url=url,
            config=crawl_config
        )

        if result.success:
            if result.extracted_content:
                data = json.loads(result.extracted_content)
                print("Extracted items:", data)
            return result[0].markdown
        else:
            print("Error:", result.error_message)


async def fetch_and_clean(url_info, sub_question):
    """Helper to fetch and clean data from a single URL"""
    if url_info[0] == 'html':
        scraped_data = await crawl4ai_func(url_info[1])
        return clean_scraped_data_llm(sub_question, scraped_data)
    else:
        ocr_response = pdf_ocr(url_info[1])
        scraped_data = "\n".join(page.markdown for page in ocr_response.pages[:10])
        return clean_scraped_data_llm(sub_question, scraped_data)