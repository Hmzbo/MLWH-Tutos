import asyncio
import sys

import argparse
import requests
import json
from typing import Tuple
from googlesearch import search
from src.web_scraper import crawl4ai_func, fetch_and_clean


async def main(url):
    if sys.platform == "win32":
        # Set the event loop policy for Windows to ensure subprocess support
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        res = await fetch_and_clean(('html', url), "How do lions communicate?")
        print('response:\n',res)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl a URL and extract content.")
    parser.add_argument("url", type=str, help="The URL to crawl.")
    args = parser.parse_args()
    asyncio.run(main(args.url))