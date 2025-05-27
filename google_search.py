import os
import requests
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")  # Your Custom Search Engine ID

def google_search(query, num_results=3):
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        raise ValueError("Missing GOOGLE_API_KEY or GOOGLE_CSE_ID environment variables.")

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": num_results,
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    results = []
    data = response.json()
    items = data.get("items", [])
    for item in items:
        title = item.get("title")
        snippet = item.get("snippet")
        link = item.get("link")
        results.append(f"{title}\n{snippet}\n{link}")

    return "\n\n".join(results)
