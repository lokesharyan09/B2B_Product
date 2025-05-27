import os
import boto3
import tiktoken
import httpx
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Body
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx
import logging

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET = os.environ.get("S3_BUCKET_NAME", "llm-customer-uploads")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set.")
if not (GOOGLE_API_KEY and GOOGLE_CSE_ID):
    logger.warning("Google Search API keys not set. Google search will not work.")
if not (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY):
    logger.warning("AWS credentials not set. S3 upload will not work.")

# Initialize OpenAI client
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    openai_client = None

# Initialize boto3 S3 client
try:
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
except Exception as e:
    logger.error(f"Failed to initialize S3 client: {e}")
    s3_client = None

router = APIRouter(prefix="/chat", tags=["chat"])

# Tokenizer for counting tokens (using tiktoken)
ENCODER = tiktoken.encoding_for_model("gpt-4")

# Constants
MAX_TOTAL_TOKENS = 8000
MAX_COMPLETION_TOKENS = 4000
MAX_PROMPT_TOKENS = MAX_TOTAL_TOKENS - MAX_COMPLETION_TOKENS

# Trigger keywords for google search augmentation
TRIGGER_KEYWORDS = [
    "inflation rate", "current inflation", "stock price", "weather today", "latest news",
    "exchange rate", "breaking news", "covid cases", "fuel price", "gold price",
    "crypto price", "sports scores", "movie release", "flight status"
]

class ChatRequest(BaseModel):
    message: str
    customer_id: Optional[str] = None
    history: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    response: str
    uploaded_files: Optional[List[str]] = None

def count_tokens(text: str) -> int:
    """Return number of tokens in text."""
    return len(ENCODER.encode(text))

def read_pdf(file_path: str) -> str:
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
    return text

def read_docx(file_path: str) -> str:
    text = ""
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        logger.error(f"Error reading DOCX: {e}")
    return text

async def google_custom_search(query: str, num_results: int = 3) -> str:
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        return "Google search is not configured."

    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": num_results,
    }
    async with httpx.AsyncClient() as client_http:
        response = await client_http.get(search_url, params=params)
        if response.status_code != 200:
            logger.error(f"Google API error: {response.status_code}")
            return f"Google Search API error: {response.status_code}"
        items = response.json().get("items", [])
        if not items:
            return "No relevant search results found."
        snippets = []
        for item in items:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            snippets.append(f"{title}\n{snippet}\n{link}\n")
        return "\n---\n".join(snippets)

async def upload_to_s3(file: UploadFile, customer_id: str) -> Optional[str]:
    if s3_client is None:
        logger.error("S3 client not initialized")
        return None
    try:
        filename = f"{customer_id}/{file.filename}"
        file_content = await file.read()
        s3_client.put_object(Bucket=S3_BUCKET, Key=filename, Body=file_content)
        return f"s3://{S3_BUCKET}/{filename}"
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        return None

@router.post("/upload", response_model=Dict[str, Any])
async def upload_file(file: UploadFile = File(...), customer_id: str = Form(...)):
    s3_path = await upload_to_s3(file, customer_id)
    if s3_path is None:
        raise HTTPException(status_code=500, detail="Failed to upload file.")
    
    # Read content for summary or indexing (based on file extension)
    local_path = f"/tmp/{file.filename}"
    with open(local_path, "wb") as f:
        f.write(await file.read())
    
    content = ""
    if file.filename.lower().endswith(".pdf"):
        content = read_pdf(local_path)
    elif file.filename.lower().endswith(".docx"):
        content = read_docx(local_path)
    else:
        content = ""

    return {
        "message": f"File uploaded to {s3_path}",
        "content_preview": content[:1000],  # Preview first 1000 chars
    }

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest = Body(...)):
    if openai_client is None:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized.")

    user_message_lower = request.message.lower()

    context_text = ""
    # Check if query needs real-time info from Google Search
    if any(keyword in user_message_lower for keyword in TRIGGER_KEYWORDS):
        google_results = await google_custom_search(request.message)
        context_text = f"Recent search results:\n{google_results}\n\n"

    # Prepare chat history with context
    messages = request.history.copy()
    messages.append({"role": "user", "content": context_text + request.message})

    # Calculate tokens and trim history if needed
    total_tokens = 0
    # We'll count tokens in messages and trim oldest if needed to keep under limit
    def tokens_in_messages(msgs: List[Dict[str, str]]) -> int:
        return sum(count_tokens(m.get("content", "")) for m in msgs) + len(msgs)*4  # rough approx tokens per message meta

    while tokens_in_messages(messages) > MAX_PROMPT_TOKENS and len(messages) > 1:
        # remove the oldest message after system prompt (if any)
        messages.pop(0)

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=MAX_COMPLETION_TOKENS,
        )
        response_text = completion.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")

    return ChatResponse(
        response=response_text,
        uploaded_files=None
    )
