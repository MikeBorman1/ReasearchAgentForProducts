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
from prompts import EXTRACT_PROMPT, MAP_PROMPT, REDUCE_PROMPT, SEARCH_QUERY_PROMPT, SEARCH_RESULT_RANK_PROMPT

def scrape(
    url: str, 
    query:str, 
    extraction_chain: LLMChain,   
    map_reduce_chain,
    map_reduce=False):
    logging.info(f'scraping {url}')
    post_url = f"https://chrome.browserless.io/content?token={constants.BROWSERLESS_API_KEY}"
    data = {
        "url": url,
    }
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }
    data_json = json.dumps(data)
    response = requests.post(post_url, headers=headers, data=data_json)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.decompose()
        text = soup.get_text()
        logging.debug(f'raw website: {text}')
        
        if len(text) > 20000 and map_reduce:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=50)
            chunks = text_splitter.create_documents([text])
            
            # running this map_reduce concurrently to call gpt api would be nice to have
            output = map_reduce_chain.run(chunks)
            logging.debug(f'extraction with mapreduce: {output}')
            return {'description': output, "url": url, "reduced": True}
        else:
            output = extraction_chain.run(text=text[:20000], query=query)
            logging.debug(f'extraction with cutoff: {output}')
            return {'description': output, "url": url, "reduced": False}
    else:
        print(f"Status: {response.status_code}, content: {response.content}")
     


async def ascrape_multiple_websites(urls: List[str], query: str, map_reduce = False):
    """
        loop throught urls, call browser to get render html, parse using bs and extract with openAI
        all are done concurrently and stream the result back to the client as it is done.
    """  
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k", api_key=constants.OPENAI_API_KEY)

    extract_prompt_template = PromptTemplate(
            template=EXTRACT_PROMPT, input_variables=["query", "text"])

    extraction_chain = LLMChain(
            llm=llm,
            prompt=extract_prompt_template,
            verbose=False,   
        )
    map_reduce_chain = load_summarize_chain(
            llm=llm,
            chain_type='map_reduce',
            verbose=False,
            map_prompt=PromptTemplate(
                template=MAP_PROMPT, 
                input_variables=['text', 'query']
                ).partial(query=query),
            combine_prompt=PromptTemplate(
                template=REDUCE_PROMPT, 
                input_variables=['text'])
        )
    
    def request(url):
        return scrape(
            url=url,
            query=query,
            extraction_chain=extraction_chain,
            map_reduce_chain=map_reduce_chain,
            map_reduce=map_reduce
        )
    
    loop = asyncio.get_event_loop()
           
    for f in asyncio.as_completed([loop.run_in_executor(None, request, url) for url in urls]):
        result = await f
        if result is not None:
            logging.info('sending result for {url}'.format(url=result.get('url')))
            yield json.dumps(result)
        else:
            yield ''

async def amulti_search(queries: List[str]):
    def single_search(query: str):
        url = "https://google.serper.dev/search"

        payload = json.dumps({
            "q": query
        })

        headers = {
            'X-API-KEY': constants.SERP_API_KEY,
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)

        return response.json()['organic'][:constants.TOP_K]
    loop = asyncio.get_event_loop()
    return await asyncio.gather(*[loop.run_in_executor(None, single_search, query) for query in queries])
   
async def allm_rank_chain(query: str):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k", api_key=constants.OPENAI_API_KEY)
    query_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(template=SEARCH_QUERY_PROMPT, input_variables=['query']),
        verbose=True
    )
    queries = json.loads(await query_chain.arun(query=query))
    search_result = await amulti_search(queries)
    rank_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            template=SEARCH_RESULT_RANK_PROMPT, 
            input_variables=['query', 'top_k', 'result']
        ).partial(query=query, top_k=constants.TOP_K),
        verbose=True
    )
    urls = await rank_chain.arun(result=search_result)
    
    return urls