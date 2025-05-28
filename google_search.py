from serpapi import GoogleSearch
import os
from dotenv import load_dotenv

load_dotenv()

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

def google_search(query, num_results=3):
    if not SERPAPI_API_KEY:
        raise ValueError("Missing SERPAPI_API_KEY environment variable.")
    
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": num_results,
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    # Extract organic results
    organic_results = results.get("organic_results", [])
    
    output = []
    for result in organic_results[:num_results]:
        title = result.get("title")
        snippet = result.get("snippet")
        link = result.get("link")
        output.append(f"{title}\n{snippet}\n{link}")

    return "\n\n".join(output)
