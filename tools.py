import json
import logging
import asyncio
import requests
from typing import List
from bs4 import BeautifulSoup

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

import constants
from prompts import (
    EXTRACT_PROMPT,
    PRICE_PROMPT,
    MAP_PROMPT,
    REDUCE_PROMPT,
    SEARCH_QUERY_PROMPT,
    SEARCH_RESULT_RANK_PROMPT,
)


def scrape(url: str, query: str, extraction_chain: LLMChain, price_chain: LLMChain):
    logging.info(f"scraping {url}")
    post_url = (
        f"https://chrome.browserless.io/content?token={constants.BROWSERLESS_API_KEY}"
    )
    data = {
        "url": url,
    }
    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "application/json",
    }
    data_json = json.dumps(data)
    response = requests.post(post_url, headers=headers, data=data_json)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.decompose()
        text = soup.get_text()
        logging.debug(f"raw website: {text}")

        output = extraction_chain.run(text=text, query=query)
        price = price_chain.run(text=text, query=query)
        return {"description": output, "price": price, "url": url, "reduced": False}
    else:
        print(f"Status: {response.status_code}, content: {response.content}")


async def ascrape_multiple_websites(urls: List[str], query: str):
    """
    loop throught urls, call browser to get render html, parse using bs and extract with openAI
    all are done concurrently and stream the result back to the client as it is done.
    """
    llm = ChatOpenAI(temperature=0, model="gpt-4o", api_key=constants.OPENAI_API_KEY)

    extraction_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            template=EXTRACT_PROMPT, input_variables=["query", "text"]
        ),
        verbose=False,
    )

    price_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(template=PRICE_PROMPT, input_variables=["query", "text"]),
        verbose=False,
    )

    def request(url):
        return scrape(
            url=url,
            query=query,
            extraction_chain=extraction_chain,
            price_chain=price_chain,
        )

    loop = asyncio.get_event_loop()

    for f in asyncio.as_completed(
        [loop.run_in_executor(None, request, url) for url in urls]
    ):
        result = await f
        if result is not None:
            logging.info("sending result for {url}".format(url=result.get("url")))
            yield json.dumps(result)
        else:
            yield json.dumps("")


async def amulti_search(queries: List[str]):
    def single_search(query: str):
        url = "https://google.serper.dev/search"

        payload = json.dumps({"q": query})

        headers = {
            "X-API-KEY": constants.SERP_API_KEY,
            "Content-Type": "application/json",
        }

        response = requests.request("POST", url, headers=headers, data=payload)

        return response.json()["organic"][: constants.TOP_K]

    loop = asyncio.get_event_loop()
    return await asyncio.gather(
        *[loop.run_in_executor(None, single_search, query) for query in queries]
    )


async def allm_rank_chain(query: str):
    llm = ChatOpenAI(
        temperature=0, model="gpt-3.5-turbo-16k", api_key=constants.OPENAI_API_KEY
    )
    query_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(template=SEARCH_QUERY_PROMPT, input_variables=["query"]),
        verbose=True,
    )
    queries = json.loads(await query_chain.arun(query=query))
    search_result = await amulti_search(queries)
    rank_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            template=SEARCH_RESULT_RANK_PROMPT,
            input_variables=["query", "top_k", "result"],
        ).partial(query=query, top_k=constants.TOP_K),
        verbose=True,
    )
    urls = await rank_chain.arun(result=search_result)

    return urls
