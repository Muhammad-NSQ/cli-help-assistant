"""
LLM Service - Handles model loading and inference
"""

import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class LLMService:
    """Service for managing LLM model loading and inference"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._loaded = False
        
    def load_model(self):
        """Load the model and tokenizer into memory"""
        try:
            model_path = self.config.model_path
            logger.info(f"Loading model from {model_path}")
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # Ensure pad token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Move to GPU if available and not using device_map
            if torch.cuda.is_available() and not hasattr(self.model, 'hf_device_map'):
                self.model = self.model.to('cuda')
                
            self.model.eval()
            self._loaded = True
            
            # Log model info
            device = next(self.model.parameters()).device
            logger.info(f"Model loaded successfully on device: {device}")
            
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"GPU memory allocated: {memory_allocated:.2f} GB")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._loaded = False
            raise
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 512, 
        temperature: float = 0.1,
        top_p: float = 0.9,
        stop_sequences: List[str] = None
    ) -> Dict:
        """Generate text completion"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=self.config.max_input_length
            )
            
            # Move to same device as model
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            
            prompt_tokens = inputs.shape[1]
            
            # Generation parameters
            generation_kwargs = {
                'max_new_tokens': max_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'do_sample': temperature > 0,
                'pad_token_id': self.tokenizer.eos_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'repetition_penalty': 1.1,
            }
            
            # Add stop sequences if provided
            if stop_sequences:
                stop_token_ids = []
                for seq in stop_sequences:
                    tokens = self.tokenizer.encode(seq, add_special_tokens=False)
                    if tokens:
                        stop_token_ids.extend(tokens)
                if stop_token_ids:
                    generation_kwargs['eos_token_id'] = stop_token_ids
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    **generation_kwargs
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove original prompt)
            generated_text = full_response[len(prompt):].strip()
            
            # Count tokens
            completion_tokens = outputs.shape[1] - prompt_tokens
            total_tokens = prompt_tokens + completion_tokens
            
            return {
                'text': generated_text,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._loaded and self.model is not None and self.tokenizer is not None
    
    def gpu_available(self) -> bool:
        """Check if GPU is available"""
        return torch.cuda.is_available()
    
    def get_device(self) -> Optional[str]:
        """Get the device the model is running on"""
        if not self.is_loaded():
            return None
        return str(next(self.model.parameters()).device)
    
    def get_memory_usage(self) -> Dict:
        """Get current memory usage"""
        info = {}
        
        if torch.cuda.is_available():
            info['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3
            info['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3
            info['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return info
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up model resources...")
        
        if self.model is not None:
            del self.model
            self.model = None
            
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self._loaded = False
        logger.info("Cleanup completed")