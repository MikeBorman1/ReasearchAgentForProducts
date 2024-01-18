# Web scraper using LLM
## How to run
1. Create a virtual environment `python -m venv .venv` and activate it `source .venv/bin/activate`
2. Create `.env` file and add secret keys
```
BROWSERLESS_API_KEY = 
SERP_API_KEY = 
OPENAI_API_KEY = 
```
3. Install dependencies. `pip install -r requirements.txt`
4. Run the server. `python main.py`
5. Go to `localhost:8000/docs` for openAPI documentation.

## API Documentation
### `get` `/search`
#### Request
1. `query`: (`string *required`) a search query to be made
2. `stream`: (`bool optional`)(`default: false`) should the server stream the scraped result or not

#### Response
List/Stream of dictionary
1. `description`: (`string`) Extracted description of the website about the product
2. `url`: (`string`) Source url of the website
3. `reduced`: (`bool`) Whether the description is extracted using map-reduced or not.
