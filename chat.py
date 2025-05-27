from fastapi import APIRouter, HTTPException, Body, UploadFile, File, Form, Query
from pydantic import BaseModel
from openai import OpenAI
import os
import boto3
import io
import json
from typing import List, Optional
from dotenv import load_dotenv
from google_search import google_search
import re

# Load environment variables
load_dotenv()

# Get API key with better error handling
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("WARNING: OPENAI_API_KEY environment variable not found!")
    print("API calls will fail. Please check your .env file.")

# Set up OpenAI API client
try:
    client = OpenAI(api_key=api_key)
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None

# S3 setup
S3_BUCKET = os.environ.get("S3_BUCKET_NAME", "llm-customer-uploads")
try:
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_REGION", "us-east-1")
    )
except Exception as e:
    print(f"Error initializing S3 client: {e}")
    s3_client = None

# Create router
router = APIRouter(prefix="/chat")

class ChatRequest(BaseModel):
    message: str
    customer_id: Optional[str] = None
    history: list = []

class ChatResponse(BaseModel):
    response: str
    uploaded_files: Optional[List[str]] = None

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(
    message: str = Form(...),
    customer_id: Optional[str] = Form(None),
    history: str = Form("[]"),  # JSON stringified history
    files: List[UploadFile] = File([])  # Optional files
):
    """
    Chat with OpenAI API with optional file upload and Google Search support.
    """
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="OpenAI client not initialized. Please check your OPENAI_API_KEY environment variable."
        )

    try:
        try:
            parsed_history = json.loads(history)
        except json.JSONDecodeError:
            parsed_history = []

        file_references = []
        uploaded_file_paths = []

        if files and customer_id:
            if s3_client is None:
                raise HTTPException(
                    status_code=500,
                    detail="S3 client not initialized. Cannot upload files."
                )

            chat_folder = f"{customer_id}/chat_files/"

            for file in files:
                file_content = await file.read()
                file_name = file.filename

                s3_key = f"{chat_folder}{file_name}"

                s3_client.put_object(
                    Bucket=S3_BUCKET,
                    Key=s3_key,
                    Body=file_content
                )

                file_references.append(f"File uploaded: {file_name}")
                uploaded_file_paths.append(s3_key)
        elif files and not customer_id:
            raise HTTPException(
                status_code=400,
                detail="customer_id is required when uploading files"
            )

        # Combine file references with user message
        user_content = message
        if file_references:
            user_content += "\n\nFiles uploaded:\n" + "\n".join(file_references)

        # Add real-time Google search if relevant
        realtime_keywords = ["inflation", "gold price", "weather", "stock", "news", "rate today", "headline", "market"]
        if any(keyword in message.lower() for keyword in realtime_keywords):
            try:
                search_results = google_search(message)
                user_content = f"Real-time search results:\n{search_results}\n\nUser asked: {user_content}"
            except Exception as search_err:
                user_content += "\n(Note: Google Search failed. Proceeding without real-time info.)"

        messages = parsed_history
        messages.append({"role": "user", "content": user_content})

        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=8000
        )

        return ChatResponse(
            response=response.choices[0].message.content,
            uploaded_files=uploaded_file_paths if uploaded_file_paths else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with OpenAI API: {str(e)}")

@router.get("/files/{customer_id}")
async def list_chat_files(customer_id: str):
    if s3_client is None:
        raise HTTPException(
            status_code=500,
            detail="S3 client not initialized. Cannot list files."
        )

    try:
        prefix = f"{customer_id}/chat_files/"
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)

        if 'Contents' not in response:
            return {
                "message": f"No chat files found for customer: {customer_id}",
                "files": []
            }

        files = [item['Key'] for item in response.get('Contents', [])]
        file_details = []

        for key in files:
            filename = key.split('/')[-1]
            file_details.append({
                "full_path": key,
                "filename": filename
            })

        return {
            "message": f"Chat files for customer: {customer_id}",
            "files": file_details
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing chat files: {str(e)}")

@router.delete("/files/{customer_id}/{file_name}")
async def delete_chat_file(customer_id: str, file_name: str):
    if s3_client is None:
        raise HTTPException(
            status_code=500,
            detail="S3 client not initialized. Cannot delete file."
        )

    try:
        s3_key = f"{customer_id}/chat_files/{file_name}"

        try:
            s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                raise HTTPException(
                    status_code=404, 
                    detail=f"Chat file not found: {file_name} for customer: {customer_id}"
                )
            else:
                raise

        s3_client.delete_object(
            Bucket=S3_BUCKET,
            Key=s3_key
        )

        return {
            "message": "Chat file deleted successfully",
            "customer_id": customer_id,
            "deleted_file": s3_key
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error deleting chat file: {str(e)}"
        )
