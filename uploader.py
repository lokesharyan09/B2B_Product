from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List, Optional
import boto3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create router
router = APIRouter(prefix="/upload")

# S3 setup
S3_BUCKET = os.environ.get("S3_BUCKET_NAME", "llm-customer-uploads")
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region_name=os.environ.get("AWS_REGION", "us-east-1")
)

@router.post("/files/")
async def upload_files(
    customer_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Upload any type of files (PDF, CSV, Word, Text, etc.) for a specific customer
    
    - **customer_id**: Unique identifier for the customer
    - **files**: List of files to upload
    """
    try:
        uploaded_files = []

        for file in files:
            file_content = await file.read()
            file_name = file.filename

            # Save under: customer_id/filename
            s3_key = f"{customer_id}/{file_name}"

            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=s3_key,
                Body=file_content
            )

            uploaded_files.append(s3_key)

        return {
            "message": "Files uploaded successfully",
            "customer_id": customer_id,
            "uploaded_files": uploaded_files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading files: {str(e)}")

@router.post("/prompt/")
async def upload_prompt(
    customer_id: str = Form(...),
    prompt_file: UploadFile = File(...)
):
    """
    Upload a prompt.txt file for a specific customer
    
    - **customer_id**: Unique identifier for the customer
    - **prompt_file**: The prompt.txt file containing instructions
    """
    try:
        if not prompt_file.filename.endswith(".txt"):
            raise HTTPException(status_code=400, detail="Only text files with .txt extension are allowed for prompts")

        file_content = await prompt_file.read()
        s3_key = f"{customer_id}/prompt.txt"

        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=file_content
        )

        return {
            "message": "Prompt file uploaded successfully",
            "customer_id": customer_id,
            "file_path": s3_key
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading prompt file: {str(e)}")

@router.get("/files/{customer_id}")
async def list_customer_files(customer_id: str):
    """
    List all files uploaded for a specific customer
    
    - **customer_id**: Unique identifier for the customer
    """
    try:
        prefix = f"{customer_id}/"
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        
        if 'Contents' not in response:
            return {
                "message": f"No files found for customer: {customer_id}",
                "files": []
            }
            
        files = [item['Key'] for item in response.get('Contents', [])]
        
        return {
            "message": f"Files for customer: {customer_id}",
            "files": files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")
    
@router.delete("/files/{customer_id}/{file_name}")
async def delete_customer_file(customer_id: str, file_name: str):
    """
    Delete a specific file for a customer
    
    - **customer_id**: Unique identifier for the customer
    - **file_name**: Name of the file to delete
    """
    try:
        # Construct the S3 key using the customer_id and file_name
        s3_key = f"{customer_id}/{file_name}"
        
        # Check if the file exists before attempting to delete
        try:
            s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                raise HTTPException(
                    status_code=404, 
                    detail=f"File not found: {file_name} for customer: {customer_id}"
                )
            else:
                raise
        
        # Delete the file from S3
        s3_client.delete_object(
            Bucket=S3_BUCKET,
            Key=s3_key
        )
        
        return {
            "message": "File deleted successfully",
            "customer_id": customer_id,
            "deleted_file": s3_key
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error deleting file: {str(e)}"
        )