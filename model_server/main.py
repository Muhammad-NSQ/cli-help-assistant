#!/usr/bin/env python3
"""
Model Server - FastAPI service for persistent LLM inference with RAG
"""

import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import List, Optional

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_server.llm_service import LLMService
from model_server.rag_service import RAGService
from model_server.config import ModelServerConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/model-server.log')
    ]
)
logger = logging.getLogger(__name__)

# Global service instances
llm_service = None
rag_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - load model and embeddings on startup"""
    global llm_service, rag_service
    
    logger.info("Starting model server...")
    config = ModelServerConfig()
    
    try:
        # Load LLM
        llm_service = LLMService(config)
        llm_service.load_model()
        logger.info("LLM loaded successfully")
        
        # Load RAG service (embeddings + knowledge base)
        rag_service = RAGService(config)
        rag_service.load()
        logger.info("RAG service loaded successfully")
        
        yield
    except Exception as e:
        logger.error(f"Failed to load services: {e}")
        sys.exit(1)
    finally:
        logger.info("Shutting down model server...")
        if llm_service:
            llm_service.cleanup()

# Create FastAPI app
app = FastAPI(
    title="CLI Help Assistant Model Server",
    description="LLM + RAG inference server for CLI help assistant",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []

class ExplainRequest(BaseModel):
    command: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    rag_loaded: bool
    gpu_available: bool
    embeddings_count: int

class ToolsResponse(BaseModel):
    tools: List[str]

class ExamplesResponse(BaseModel):
    examples: List[str]

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global llm_service, rag_service
    
    return HealthResponse(
        status="healthy" if (llm_service and llm_service.is_loaded() and rag_service and rag_service.is_loaded()) else "unhealthy",
        model_loaded=llm_service.is_loaded() if llm_service else False,
        rag_loaded=rag_service.is_loaded() if rag_service else False,
        gpu_available=llm_service.gpu_available() if llm_service else False,
        embeddings_count=len(rag_service.text_chunks) if rag_service and rag_service.is_loaded() else 0
    )

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Answer a query using RAG + LLM"""
    global llm_service, rag_service
    
    if not llm_service or not llm_service.is_loaded():
        raise HTTPException(status_code=503, detail="LLM not loaded")
    
    if not rag_service or not rag_service.is_loaded():
        raise HTTPException(status_code=503, detail="RAG service not loaded")
    
    try:
        result = rag_service.query(request.query, top_k=request.top_k)
        
        if not result['relevant_info']:
            return QueryResponse(
                answer=f"I don't have information about '{request.query}'. Try /tools to see available commands.",
                sources=[]
            )
        
        # Build prompt
        context = "\n\n".join([info.get('text', '') for info in result['relevant_info'][:3]])
        prompt = f"""Context:
{context}

Question: {request.query}

Provide a clear, concise answer using the context above.

Answer:"""
        
        # Generate response
        llm_result = llm_service.generate(
            prompt=prompt,
            max_tokens=150,
            temperature=0.0,
            top_p=0.9
        )
        
        response_text = llm_result['text']
        
        # Clean response
        response_text = rag_service.clean_response(response_text)
        
        # Get related commands
        sources = result['related_commands'][:3]
        
        return QueryResponse(
            answer=response_text,
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/explain", response_model=QueryResponse)
async def explain(request: ExplainRequest):
    """Explain a specific command"""
    global llm_service, rag_service
    
    if not llm_service or not llm_service.is_loaded():
        raise HTTPException(status_code=503, detail="LLM not loaded")
    
    if not rag_service or not rag_service.is_loaded():
        raise HTTPException(status_code=503, detail="RAG service not loaded")
    
    try:
        result = rag_service.search_command(request.command)
        
        if not result['relevant_info']:
            return QueryResponse(
                answer=f"I don't have information about '{request.command}'.",
                sources=[]
            )
        
        context = "\n\n".join([info.get('text', '') for info in result['relevant_info'][:2]])
        prompt = f"Explain the '{request.command}' command:\n\n{context}\n\nExplanation:"
        
        llm_result = llm_service.generate(
            prompt=prompt,
            max_tokens=150,
            temperature=0.0
        )
        
        response_text = rag_service.clean_response(llm_result['text'])
        
        return QueryResponse(
            answer=response_text,
            sources=[request.command]
        )
        
    except Exception as e:
        logger.error(f"Explain error: {e}")
        raise HTTPException(status_code=500, detail=f"Explain failed: {str(e)}")

@app.get("/tools", response_model=ToolsResponse)
async def list_tools():
    """List all available tools"""
    global rag_service
    
    if not rag_service or not rag_service.is_loaded():
        raise HTTPException(status_code=503, detail="RAG service not loaded")
    
    try:
        tools = rag_service.list_tools()
        return ToolsResponse(tools=tools)
    except Exception as e:
        logger.error(f"List tools error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools/{tool}/examples", response_model=ExamplesResponse)
async def get_examples(tool: str):
    """Get examples for a specific tool"""
    global rag_service
    
    if not rag_service or not rag_service.is_loaded():
        raise HTTPException(status_code=503, detail="RAG service not loaded")
    
    try:
        examples = rag_service.get_examples(tool)
        return ExamplesResponse(examples=examples)
    except Exception as e:
        logger.error(f"Get examples error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rebuild-knowledge")
async def rebuild_knowledge():
    """Rebuild the knowledge base and embeddings"""
    global rag_service
    
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    
    try:
        rag_service.rebuild()
        return {"status": "success", "message": "Knowledge base rebuilt successfully"}
    except Exception as e:
        logger.error(f"Rebuild error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Run server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )