# Product Recommendation System

A FastAPI application that provides product recommendations based on customer data, product information, and industry specifications from CSV files stored in Amazon S3. It uses OpenAI's GPT-4 API to generate recommendations based on user-provided prompts.

## Features

- Upload CSV files and prompts to Amazon S3
- Generate product recommendations based on customer-specific data
- Interactive chat with OpenAI's GPT-4
- API endpoints for file management, recommendations, and chat

## Project Structure

```
├── app.py              # Main FastAPI application
├── uploader.py         # File upload functionality
├── recommender.py      # Product recommendation logic
├── chat.py             # OpenAI chat interaction
├── requirements.txt    # Python dependencies
├── Dockerfile          # For containerization
└── .env                # Environment variables
```

## API Endpoints

### File Upload Endpoints

- `POST /upload/files/`: Upload CSV files for a customer
- `POST /upload/prompt/`: Upload a prompt.txt file for a customer
- `GET /upload/files/{customer_id}`: List all files for a customer

### Recommendation Endpoints

- `POST /recommend/products`: Generate product recommendations
- `GET /recommend/industries/{customer_id}`: List available industries for a customer

### Chat Endpoints

- `POST /chat/`: Chat with OpenAI API
- `POST /chat/stream`: Stream chat responses from OpenAI API

## Setup and Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Set up environment variables

Create a `.env` file in the project root with the following variables:

```
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=llm-customer-uploads
OPENAI_API_KEY=your_openai_api_key
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application locally

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000` and the API documentation at `http://localhost:8000/docs`.

## Deployment to Render

This application is configured to be deployed on Render.com. To deploy:

1. Create a new Web Service in Render
2. Connect your GitHub repository
3. Configure the following settings:
   - **Environment**: Docker
   - **Build Command**: (leave empty, will use Dockerfile)
   - **Start Command**: (leave empty, will use CMD in Dockerfile)
   - **Environment Variables**: Add all the variables from `.env`

## Docker Deployment

The application can also be deployed using Docker:

```bash
docker build -t product-recommendation-api .
docker run -p 8000:8000 -e PORT=8000 --env-file .env product-recommendation-api
```

## S3 File Structure

The application expects the following file structure in S3:

```
customer_id/
  ├── base.csv           # Base product information
  ├── healthcare.csv     # Industry-specific product information
  ├── finance.csv        # Industry-specific product information
  ├── prompt.txt         # Instructions for the LLM
  └── ...
```

## Usage Examples

### Uploading Files

```bash
curl -X POST "http://localhost:8000/upload/files/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "customer_id=customer123" \
  -F "files=@base.csv" \
  -F "files=@healthcare.csv"
```

### Getting Recommendations

```bash
curl -X POST "http://localhost:8000/recommend/products" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "customer123",
    "products": [
      {
        "productName": "ProductA",
        "industry": "Healthcare"
      }
    ]
  }'
```