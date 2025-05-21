from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import boto3
import io
import json
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Get OpenAI API key with error handling
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("WARNING: OPENAI_API_KEY environment variable not found in recommender.py!")
    print("API calls will fail. Please check your .env file.")

# Set up OpenAI client
try:
    client = OpenAI(api_key=api_key)
except Exception as e:
    print(f"Error initializing OpenAI client in recommender.py: {e}")
    client = None

# Create router
router = APIRouter(prefix="/recommend")

# Get AWS credentials with error handling
aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
aws_region = os.environ.get('AWS_REGION', 'us-east-1')

if not aws_access_key or not aws_secret_key:
    print("WARNING: AWS credentials not found! S3 operations will fail.")
    print("Please check your .env file for AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")

# Initialize S3 client
try:
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )
except Exception as e:
    print(f"Error initializing S3 client: {e}")
    s3 = None

# Configuration
BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', "llm-customer-uploads")

# Define request models
class Product(BaseModel):
    productName: str
    industry: str

class RecommendationRequest(BaseModel):
    customer_id: str
    products: List[Product] = Field(..., min_items=1, max_items=50)

# Helper functions
def read_csv_from_s3(key):
    """Read CSV file from S3 and return as Pandas DataFrame"""
    if s3 is None:
        raise HTTPException(status_code=500, detail="S3 client not initialized. Check AWS credentials.")
    
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    return pd.read_csv(io.BytesIO(obj['Body'].read()))

def read_text_from_s3(key):
    """Read text file from S3 and return as string"""
    if s3 is None:
        raise HTTPException(status_code=500, detail="S3 client not initialized. Check AWS credentials.")
    
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    return obj['Body'].read().decode('utf-8')

def format_sample_data_for_prompt(base_df, industry_dfs, products):
    """Format product data for the LLM prompt with flexibility for different schemas"""
    examples = []

    for product in products:
        name = product.productName
        industry = product.industry

        # Find the product in the base dataframe using flexible matching
        base_match = None
        
        # First, try exact match on first column
        exact_match = base_df[base_df.iloc[:, 0].astype(str).str.strip().str.lower() == name.strip().lower()]
        if not exact_match.empty:
            base_match = exact_match.iloc[0]
        else:
            # Try partial match instead
            partial_match = base_df[base_df.iloc[:, 0].astype(str).str.strip().str.lower().str.contains(name.strip().lower())]
            if not partial_match.empty:
                base_match = partial_match.iloc[0]
        
        if base_match is None:
            continue
        
        # Find the corresponding industry dataframe
        industry_df = None
        industry_match = None
        
        for ind_key, ind_df in industry_dfs.items():
            if ind_key.lower() == industry.lower():
                industry_df = ind_df
                
                # Look for the product in this industry dataframe
                # Use flexible matching
                product_match = ind_df[ind_df[ind_df.columns[0]].astype(str).str.lower().str.contains(name.strip().lower())]
                if not product_match.empty:
                    industry_match = product_match.iloc[0]
                    break
        
        if industry_df is None or industry_match is None:
            continue

        # Dynamically build example based on available columns
        example = f"""
Product: {name}
Industry: {industry}

Base:
"""
        # Dynamically add base product information
        for col in base_df.columns:
            if pd.notna(base_match[col]):
                example += f"- {col}: {base_match[col]}\n"
                
        example += "\nIndustry Variant:\n"
        
        # Dynamically add industry variant information
        for col in industry_df.columns:
            if pd.notna(industry_match[col]):
                example += f"- {col}: {industry_match[col]}\n"
                
        examples.append(example)

    return "\n---\n".join(examples)

@router.post("/products")
async def recommend_products(request: RecommendationRequest):
    """
    Generate product recommendations based on customer data and industry specifications
    
    - **customer_id**: Unique identifier for the customer
    - **products**: List of products with their industries for recommendation
    """
    # Check if clients are initialized properly
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="OpenAI client not initialized. Please check your OPENAI_API_KEY environment variable."
        )
    
    if s3 is None:
        raise HTTPException(
            status_code=500,
            detail="S3 client not initialized. Please check your AWS credentials."
        )
    
    try:
        customer_id = request.customer_id
        products = request.products

        # List files in S3 for the customer
        prefix = f"{customer_id}/"
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
        
        if 'Contents' not in response:
            raise HTTPException(status_code=404, detail=f"No files found for customer_id: {customer_id}")
            
        files = [item['Key'] for item in response.get('Contents', []) if item['Key'].endswith('.csv')]
        
        if not files:
            raise HTTPException(status_code=404, detail=f"No CSV files found for customer_id: {customer_id}")

        base_df = None
        industry_dfs = {}

        # Identify the base file and industry files
        for file_key in files:
            filename = os.path.basename(file_key).lower()
            
            if "base" in filename:
                base_df = read_csv_from_s3(file_key)
            else:
                # For non-base files, extract the industry name from the filename
                df = read_csv_from_s3(file_key)
                
                # Extract potential industry name from the filename
                industry_name = filename.split('_')[0].capitalize()
                if industry_name:
                    # Remove .csv extension if present
                    industry_name = industry_name.replace(".csv", "")
                    industry_dfs[industry_name] = df

        if base_df is None:
            # Try to use the first CSV file as base if none is explicitly marked
            for file_key in files:
                if file_key.endswith('.csv'):
                    base_df = read_csv_from_s3(file_key)
                    break
                
        if base_df is None:
            raise HTTPException(
                status_code=400, 
                detail="Base file not found in folder. Please ensure there is a CSV file with 'base' in the name or at least one CSV file."
            )
            
        if not industry_dfs:
            # Use all non-base CSVs as industry files
            for file_key in files:
                if "base" not in os.path.basename(file_key).lower() and file_key.endswith('.csv'):
                    filename = os.path.basename(file_key).lower()
                    industry_name = filename.split('_')[0].capitalize()
                    # Remove .csv extension if present
                    industry_name = industry_name.replace(".csv", "")
                    industry_dfs[industry_name] = read_csv_from_s3(file_key)
                
        if not industry_dfs:
            raise HTTPException(
                status_code=400, 
                detail="No industry files found. Please upload CSV files for relevant industries."
            )
            
        # Log available industries
        available_industries = list(industry_dfs.keys())
        print(f"Available industries: {available_industries}")
            
        # Validate that the requested industries exist in available files
        missing_industries = []
        for product in products:
            industry = product.industry
            if industry and not any(ind.lower() == industry.lower() for ind in industry_dfs.keys()):
                missing_industries.append(industry)
                
        if missing_industries:
            unique_missing = list(set(missing_industries))
            raise HTTPException(
                status_code=400,
                detail=f"Some requested industries are not available: {', '.join(unique_missing)}. Available industries: {', '.join(available_industries)}"
            )

        prompt_data = format_sample_data_for_prompt(base_df, industry_dfs, products)
        
        if not prompt_data:
            raise HTTPException(
                status_code=400,
                detail="Could not match any products with base and industry data. Please check product names and industries."
            )
        
        # Load the prompt from S3
        prompt_file_key = f"{prefix}prompt.txt"
        try:
            base_prompt = read_text_from_s3(prompt_file_key)
        except Exception as e:
            # If no prompt.txt exists, use a default prompt
            base_prompt = """Please analyze the provided product data across different industries and provide recommendations.
Based on the information given, generate insights about how these products could be optimized for their specific industries.
Return your response as a JSON object with an array of recommendations for each product."""
            
        # Combine the base prompt with the dynamically generated data
        full_prompt = f"{base_prompt.strip()}\n\nData:\n{prompt_data}"

        # Call OpenAI API
        gpt_response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.4
        )

        try:
            gpt_text = gpt_response.choices[0].message.content
            
            # Try to parse as JSON first
            try:
                json_result = json.loads(gpt_text)
                
                return {
                    "message": "Processed successfully",
                    "availableIndustries": list(industry_dfs.keys()),
                    "matchedProducts": len(prompt_data.split("---")),
                    "gptResponse": json_result
                }
            except json.JSONDecodeError:
                # If not valid JSON, return the text response directly
                return {
                    "message": "Processed successfully",
                    "availableIndustries": list(industry_dfs.keys()),
                    "matchedProducts": len(prompt_data.split("---")),
                    "gptResponse": gpt_text
                }
        except Exception as e:
            # Handle case where GPT doesn't return valid JSON
            raise HTTPException(
                status_code=422,
                detail={
                    "error": f"Error processing response: {str(e)}",
                    "rawResponse": gpt_text if 'gpt_text' in locals() else "No response generated"
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/industries/{customer_id}")
async def list_available_industries(customer_id: str):
    """
    List all available industries for a specific customer
    
    - **customer_id**: Unique identifier for the customer
    """
    # Check if S3 client is initialized properly
    if s3 is None:
        raise HTTPException(
            status_code=500,
            detail="S3 client not initialized. Please check your AWS credentials."
        )
    
    try:
        # List files in S3 for the customer
        prefix = f"{customer_id}/"
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
        
        if 'Contents' not in response:
            raise HTTPException(status_code=404, detail=f"No files found for customer_id: {customer_id}")
            
        files = [item['Key'] for item in response.get('Contents', []) if item['Key'].endswith('.csv')]
        
        if not files:
            raise HTTPException(status_code=404, detail=f"No CSV files found for customer_id: {customer_id}")

        # Extract industry names from filenames
        industries = []
        for file_key in files:
            filename = os.path.basename(file_key).lower()
            
            if "base" not in filename:
                # Extract potential industry name from the filename
                industry_name = filename.split('_')[0].capitalize()
                # Remove .csv extension if present
                industry_name = industry_name.replace(".csv", "")
                if industry_name and industry_name not in industries:
                    industries.append(industry_name)

        return {
            "customer_id": customer_id,
            "industries": industries
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))