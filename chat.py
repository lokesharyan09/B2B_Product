from fastapi import APIRouter, HTTPException, Body, UploadFile, File, Form
from pydantic import BaseModel
from openai import OpenAI
import os
import boto3
import json
import requests
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
google_api_key = os.environ.get("GOOGLE_API_KEY")          # Google Cloud API key
google_cse_id = os.environ.get("GOOGLE_CSE_ID")            # Google Custom Search Engine ID
S3_BUCKET = os.environ.get("S3_BUCKET_NAME", "llm-customer-uploads")

try:
    client = OpenAI(api_key=api_key)
except Exception as e:
    print(f"OpenAI client init error: {e}")
    client = None

try:
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_REGION", "us-east-1")
    )
except Exception as e:
    print(f"S3 client init error: {e}")
    s3_client = None

router = APIRouter(prefix="/chat")

class ChatRequest(BaseModel):
    message: str
    customer_id: Optional[str] = None
    history: list = []

class ChatResponse(BaseModel):
    response: str
    uploaded_files: Optional[List[str]] = None

def perform_google_search_google_api(query: str) -> str:
    if not google_api_key or not google_cse_id:
        return "Google Search API key or CSE ID not configured."

    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": google_api_key,
        "cx": google_cse_id,
        "q": query,
        "num": 3,
    }
    try:
        res = requests.get(search_url, params=params)
        res.raise_for_status()
        data = res.json()
        snippets = []
        for item in data.get("items", []):
            snippet = item.get("snippet")
            if snippet:
                snippets.append(snippet)
        return "\n".join(snippets) if snippets else "No relevant information found."
    except Exception as e:
        return f"Search error: {str(e)}"

def trim_messages_to_fit(messages, max_tokens=8000, completion_tokens=1000):
    approx_token_limit = max_tokens - completion_tokens
    total_tokens = sum(len(m["content"]) // 4 + 4 for m in messages)
    while total_tokens > approx_token_limit and messages:
        messages.pop(0)
        total_tokens = sum(len(m["content"]) // 4 + 4 for m in messages)
    return messages

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest = Body(...)):
    if client is None:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized.")

    messages = request.history or []

    user_message_lower = request.message.lower()
    trigger_keywords = ["inflation rate", "current inflation", "inflation today", "today's inflation", "inflation update"]

    if any(keyword in user_message_lower for keyword in trigger_keywords):
        search_results = perform_google_search_google_api(request.message)
        augmented_content = (
            "Use the following information from the web to answer the question accurately:\n"
            f"{search_results}\n\nQuestion: {request.message}"
        )
        messages.append({"role": "user", "content": augmented_content})
    else:
        messages.append({"role": "user", "content": request.message})

    messages = trim_messages_to_fit(messages, max_tokens=8000, completion_tokens=1000)

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        return ChatResponse(
            response=response.choices[0].message.content,
            uploaded_files=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with OpenAI API: {str(e)}")

@router.post("/with-files", response_model=ChatResponse)
async def chat_with_files_endpoint(
    message: str = Form(...),
    customer_id: str = Form(...),
    history: str = Form("[]"),
    files: List[UploadFile] = File(...)
):
    if client is None:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized.")
    if s3_client is None:
        raise HTTPException(status_code=500, detail="S3 client not initialized.")

    try:
        parsed_history = json.loads(history)
    except json.JSONDecodeError:
        parsed_history = []

    file_references = []
    uploaded_file_paths = []
    chat_folder = f"{customer_id}/chat_files/"

    for file in files:
        if file.filename and file.filename.strip():
            file_content = await file.read()
            s3_key = f"{chat_folder}{file.filename}"
            s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=file_content)
            file_references.append(f"File uploaded: {file.filename}")
            uploaded_file_paths.append(s3_key)

    user_content = message
    if file_references:
        user_content += "\n\nFiles uploaded:\n" + "\n".join(file_references)

    messages = parsed_history or []

    user_message_lower = message.lower()
    trigger_keywords = ["inflation rate", "current inflation", "inflation today", "today's inflation", "inflation update"]

    if any(keyword in user_message_lower for keyword in trigger_keywords):
        search_results = perform_google_search_google_api(message)
        augmented_content = (
            "Use the following information from the web to answer the question accurately:\n"
            f"{search_results}\n\nQuestion: {user_content}"
        )
        messages.append({"role": "user", "content": augmented_content})
    else:
        messages.append({"role": "user", "content": user_content})

    messages = trim_messages_to_fit(messages, max_tokens=8000, completion_tokens=1000)

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        return ChatResponse(
            response=response.choices[0].message.content,
            uploaded_files=uploaded_file_paths
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with OpenAI API: {str(e)}")

@router.get("/files/{customer_id}")
async def list_chat_files(customer_id: str):
    if s3_client is None:
        raise HTTPException(status_code=500, detail="S3 client not initialized.")

    try:
        prefix = f"{customer_id}/chat_files/"
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)

        if 'Contents' not in response:
            return {
                "message": f"No chat files found for customer: {customer_id}",
                "files": []
            }

        files = [item['Key'] for item in response.get('Contents', [])]
        file_details = [{"full_path": key, "filename": key.split('/')[-1]} for key in files]

        return {
            "message": f"Chat files for customer: {customer_id}",
            "files": file_details
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing chat files: {str(e)}")

@router.delete("/files/{customer_id}/{file_name}")
async def delete_chat_file(customer_id: str, file_name: str):
    if s3_client is None:
        raise HTTPException(status_code=500, detail="S3 client not initialized.")

    try:
        s3_key = f"{customer_id}/chat_files/{file_name}"

        try:
            s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                raise HTTPException(status_code=404, detail=f"Chat file not found: {file_name}")
            else:
                raise

        s3_client.delete_object(Bucket=S3_BUCKET, Key=s3_key)

        return {
            "message": "Chat file deleted successfully",
            "customer_id": customer_id,
            "deleted_file": s3_key
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting chat file: {str(e)}")
