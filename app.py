from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import routers from other modules
from uploader import router as uploader_router
from chat import router as chat_router
from recommender import router as recommender_router

# Create FastAPI app
app = FastAPI(
    title="Product Recommendation System",
    description="API for product recommendations using LLM and customer-specific data",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers from other modules
app.include_router(uploader_router, tags=["File Upload"])
app.include_router(recommender_router, tags=["Product Recommendations"])
app.include_router(chat_router, tags=["Chat"])

# Root endpoint for health checks
@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "Product Recommendation API is running"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)