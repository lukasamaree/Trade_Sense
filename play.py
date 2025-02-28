from playwright.sync_api import sync_playwright
import time
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
import os


elapsed = time.time()

os.system("playwright install chromium")


def retrieve_urls_for_docs(ticker):
    url = "https://finance.yahoo.com/quote/" + ticker + "/news/"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # Consider headless mode for faster performance
        page = browser.new_page()

        # Increase timeout and use wait_until
        page.goto(url, timeout=60000, wait_until="domcontentloaded")

        # Optionally, block images to speed up loading
        page.route("**/*.{png,jpg,jpeg,gif}", lambda route: route.abort())
        
        # Locate company article
        ad = page.locator("div.publishing.yf-1weyqlp")
        texts = ad.evaluate_all("elements => elements.map(el => el.textContent.trim())")
        texts = [text.split(" • ")[0] for text in texts]
        


        page.wait_for_selector("a.subtle-link.fin-size-small.thumb.yf-1xqzjha", timeout=60000)

        # Locate all the anchor tags and extract href attributes

        links = page.locator("a.subtle-link.fin-size-small.thumb.yf-1xqzjha")
        hrefs = links.evaluate_all("elements => elements.map(el => el.href)")
        browser.close()
        print(f'this is elapsed time: {time.time() - elapsed}')
        
        # Print extracted links
        return hrefs[:5]





