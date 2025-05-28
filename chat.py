from fastapi import APIRouter, HTTPException, Body, UploadFile, File, Form
from pydantic import BaseModel
from openai import OpenAI
import os
import boto3
import json
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API keys and config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "llm-customer-uploads")

# OpenAI client
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"OpenAI error: {e}")
    client = None

# S3 client
try:
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1")
    )
except Exception as e:
    print(f"S3 init error: {e}")
    s3_client = None

router = APIRouter(prefix="/chat")

class ChatRequest(BaseModel):
    message: str
    customer_id: Optional[str] = None
    history: List[dict] = []

class ChatResponse(BaseModel):
    response: str
    uploaded_files: Optional[List[str]] = None

# Tool schema for web search
tool_definitions = [{
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Perform a real-time web search using SerpAPI",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to look up"
                }
            },
            "required": ["query"]
        }
    }
}]

# Web search function
def search_web(query):
    from serpapi import GoogleSearch
    if not SERPAPI_API_KEY:
        return "Web search is unavailable: SERPAPI_API_KEY missing."
    
    search = GoogleSearch({
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": 3
    })
    results = search.get_dict()
    items = results.get("organic_results", [])
    if not items:
        return "No relevant web results found."

    response = []
    for item in items:
        title = item.get("title")
        snippet = item.get("snippet")
        link = item.get("link")
        response.append(f"**{title}**\n{snippet}\n{link}")
    return "\n\n".join(response)

def run_function_call(name, arguments):
    if name == "search_web":
        return search_web(arguments.get("query"))
    return "Unknown function call."

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest = Body(...)):
    if client is None:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")

    try:
        messages = request.history.copy()
        messages.append({"role": "user", "content": request.message})

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            tools=tool_definitions,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=2000
        )

        choice = response.choices[0]

        if choice.finish_reason == "tool_calls":
            tool_call = choice.message.tool_calls[0]
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            tool_response = run_function_call(func_name, args)

            messages.append(choice.message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_response
            })

            followup = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            return ChatResponse(response=followup.choices[0].message.content)

        return ChatResponse(response=choice.message.content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {e}")

@router.post("/with-files", response_model=ChatResponse)
async def chat_with_files(
    message: str = Form(...),
    customer_id: str = Form(...),
    history: str = Form("[]"),
    files: List[UploadFile] = File(...)
):
    if client is None or s3_client is None:
        raise HTTPException(status_code=500, detail="Client not initialized")

    try:
        parsed_history = json.loads(history)
        uploaded_paths = []
        file_notes = []

        folder = f"{customer_id}/chat_files/"
        for file in files:
            if file.filename:
                content = await file.read()
                key = f"{folder}{file.filename}"
                s3_client.put_object(Bucket=S3_BUCKET, Key=key, Body=content)
                uploaded_paths.append(key)
                file_notes.append(f"Uploaded: {file.filename}")

        user_message = f"{message}\n\n" + "\n".join(file_notes)
        parsed_history.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=parsed_history,
            temperature=0.7,
            max_tokens=2000
        )

        return ChatResponse(response=response.choices[0].message.content, uploaded_files=uploaded_paths)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat with files error: {e}")

@router.get("/files/{customer_id}")
async def list_files(customer_id: str):
    if s3_client is None:
        raise HTTPException(status_code=500, detail="S3 client not initialized")

    try:
        prefix = f"{customer_id}/chat_files/"
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        contents = response.get("Contents", [])
        return {
            "message": f"Files for {customer_id}",
            "files": [{"key": obj["Key"], "filename": obj["Key"].split('/')[-1]} for obj in contents]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {e}")

@router.delete("/files/{customer_id}/{file_name}")
async def delete_file(customer_id: str, file_name: str):
    if s3_client is None:
        raise HTTPException(status_code=500, detail="S3 client not initialized")

    try:
        key = f"{customer_id}/chat_files/{file_name}"
        s3_client.head_object(Bucket=S3_BUCKET, Key=key)
        s3_client.delete_object(Bucket=S3_BUCKET, Key=key)
        return {"message": "File deleted", "key": key}
    except s3_client.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            raise HTTPException(status_code=404, detail="File not found")
        else:
            raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")
