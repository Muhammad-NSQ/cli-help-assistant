#!/usr/bin/env python3
'''
Model Service - Dedicated service for LLM inference
Keeps model loaded in VRAM for fast responses
'''

import logging
import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.1

class GenerateResponse(BaseModel):
    text: str
    success: bool
    error: str = None

class ModelService:
    '''Singleton model service that keeps model in VRAM'''
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_loaded = False
        
    def load_model(self):
        '''Load model into VRAM once at startup'''
        if self.model_loaded:
            return
            
        try:
            model_path = os.getenv('MODEL_PATH', 'deepseek-ai/deepseek-coder-1.3b-instruct')
            
            logger.info(f"Loading model from {model_path}...")
            
            # Determine device
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info("Using CUDA device")
            else:
                self.device = "cpu"
                logger.info("Using CPU device")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model_loaded = True
            logger.info("Model loaded successfully and cached in VRAM")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.1) -> str:
        '''Generate response from prompt'''
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Encode input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part
            response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

# Global model service instance
model_service = ModelService()

# FastAPI app
app = FastAPI(title="CLI Help Model Service", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    '''Load model at startup'''
    logger.info("Starting model service...")
    model_service.load_model()
    logger.info("Model service ready")

@app.get("/health")
async def health_check():
    '''Health check endpoint'''
    return {
        "status": "healthy",
        "model_loaded": model_service.model_loaded,
        "device": model_service.device
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    '''Generate text from prompt'''
    try:
        if not model_service.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        response_text = model_service.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return GenerateResponse(
            text=response_text,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return GenerateResponse(
            text="",
            success=False,
            error=str(e)
        )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)