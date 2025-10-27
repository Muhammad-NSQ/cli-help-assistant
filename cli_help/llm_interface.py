'''Interface for LLM integration - HTTP client for model server'''

import logging
import requests
import time
from typing import Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class LLMInterface:
    '''HTTP client for model server communication'''
    
    def __init__(self, config):
        self.config = config
        self.server_url = getattr(config, 'model_server_url', 'http://localhost:8000')
        self.timeout = getattr(config, 'model_server_timeout', 30)
        self.max_retries = getattr(config, 'model_server_retries', 3)
        
        # Setup HTTP session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
        total=self.max_retries,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],  # ← NEW
        backoff_factor=1
    )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Check server health on initialization
        self._check_server_health()
    
    def _check_server_health(self) -> bool:
        """Check if model server is healthy"""
        try:
            response = self.session.get(
                f"{self.server_url}/health",
                timeout=5
            )
            response.raise_for_status()
            health_data = response.json()
            
            if health_data.get('status') == 'healthy' and health_data.get('model_loaded'):
                logger.info("Model server is healthy and ready")
                return True
            else:
                logger.warning(f"Model server unhealthy: {health_data}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Model server health check failed: {e}")
            return False
    
    def generate_response(self, prompt: str) -> str:
        '''Generate response from model server'''
        try:
            # Prepare request payload
            payload = {
                "prompt": prompt,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": getattr(self.config, 'top_p', 0.9),
                "stop_sequences": []
            }
            
            # Make request to model server
            response = self.session.post(
                f"{self.server_url}/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            generated_text = result.get('response', '')
            
            # Log token usage for debugging
            prompt_tokens = result.get('prompt_tokens', 0)
            completion_tokens = result.get('completion_tokens', 0)
            logger.debug(f"Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}")
            
            return generated_text
            
        except requests.exceptions.Timeout:
            logger.error("Model server request timed out")
            return self._fallback_response(prompt)
            
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to model server")
            return self._fallback_response(prompt)
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"Model server HTTP error: {e}")
            if e.response.status_code == 503:
                return "Model server is currently unavailable. Please try again in a moment."
            return self._fallback_response(prompt)
            
        except Exception as e:
            logger.error(f"Unexpected error communicating with model server: {e}")
            return self._fallback_response(prompt)
    
    def _fallback_response(self, prompt: str) -> str:
        '''Generate fallback response when model server is unavailable'''
        logger.info("Using fallback response")
        
        if "find" in prompt.lower():
            return '''To find files, you can use several approaches:

1. **find command**: `find . -name "*.py"` - Search for Python files
2. **ls with grep**: `ls -la | grep .py` - List and filter Python files  
3. **locate**: `locate "*.py"` - Fast search using database

The `find` command is most versatile and works on all Unix systems.

*Note: Model server unavailable - showing cached response*'''
        
        elif "git" in prompt.lower():
            return '''Git is a distributed version control system. Common commands:

- `git status` - Show current repository state
- `git add .` - Stage all changes
- `git commit -m "message"` - Commit staged changes
- `git push` - Push to remote repository

Always check `git status` before making changes.

*Note: Model server unavailable - showing cached response*'''
        
        else:
            return '''I can help explain command-line tools and their usage. 
Try asking about specific commands like "find", "git", "grep", or "ls".

For example:
- "how do I find Python files?"
- "explain git status"
- "show me grep examples"

*Note: Model server unavailable - showing cached response*'''
    
    def get_server_info(self) -> dict:
        """Get model server information"""
        try:
            response = self.session.get(
                f"{self.server_url}/model/info",
                timeout=5
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get server info: {e}")
            return {"error": str(e)}
    
    def is_server_available(self) -> bool:
        """Check if server is available"""
        return self._check_server_health()
    
    def is_loaded(self) -> bool:
        '''Check if model server is available and ready'''
        return self.is_server_available()
    
    def clear_cache(self):
        '''No-op for HTTP client - cache is managed by server'''
        pass