import json
import logging
import uvicorn
import constants
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from tools import ascrape_multiple_websites, allm_rank_chain

app = FastAPI()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)  

@app.get('/search', description="let llm figure out best query to search in google, pick top relevent site and extract")
async def llm_search(query: str, stream: Optional[bool] = False):
    """
    # let llm figure out the query
    # search those query in google
    # let llm pick and rank top k result or consolidate result to remove duplicate
    # start scraping the web
    # stream the result one by one for better user performancez
    """
    search_result = json.loads(await allm_rank_chain(query))
    gen = ascrape_multiple_websites(search_result, query, map_reduce=True)
    if stream:
        return StreamingResponse(gen, media_type='text/event-stream')
    else:
        return [json.loads(res) async for res in gen]
    
if __name__ == '__main__':
    uvicorn.run('main:app', reload=True, port=constants.PORT)