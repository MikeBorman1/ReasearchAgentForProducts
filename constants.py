import os
from dotenv import load_dotenv

load_dotenv()

BROWSERLESS_API_KEY = os.getenv('BROWSERLESS_API_KEY')
SERP_API_KEY = os.getenv('SERP_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PORT = int(os.getenv('PORT', '8000'))

TOP_K = 10