

import logging

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

import constants
from prompts import EXTRACT_PROMPT, MAP_PROMPT, REDUCE_PROMPT
from tools import scrape

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)  

url = 'https://www.vapospy.com/p/pax-plus'
query = 'pax plus'

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

output = scrape(
        url=url,
        query=query,
        extraction_chain=extraction_chain,
        map_reduce_chain=map_reduce_chain,
        map_reduce=True
    )

print(output)