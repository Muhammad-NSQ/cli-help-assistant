"""
Model Server Configuration
"""

import os
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ModelServerConfig:
    """Configuration for the model server"""
    
    def __init__(self, config_path=None):
        # Model settings
        self.model_name = "deepseek-coder"
        self.model_path = Path("./models/deepseek-coder-v2-lite")
        
        # Embedding settings
        self.embedding_model = "all-MiniLM-L6-v2"
        
        # Knowledge base paths
        self.knowledge_base_path = Path("./knowledge")
        self.embeddings_path = Path("./knowledge/processed")
        
        # Generation parameters
        self.max_tokens = 150
        self.temperature = 0.0
        self.top_p = 0.9
        self.max_input_length = 2048
        
        # Server settings
        self.host = "0.0.0.0"
        self.port = 8000
        self.workers = 1
        
        # Performance settings
        self.device = "auto"
        self.torch_dtype = "float16"
        self.trust_remote_code = True
        
        # Load from environment
        self._load_from_env()
        # self._find_available_model()
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"No model found at {self.model_path}")
    
    def _find_available_model(self):
        """Try to find an available model"""
        potential_paths = [
            Path("./models/deepseek-coder-v2-lite"),
            Path("./models/deepseek-coder-1.3b"),
            Path("./model_server/models/deepseek-coder-1.3b"),
            Path("/app/models/deepseek-coder-v2-lite"),
            Path("/app/models/deepseek-coder-1.3b"),
            Path("/app/model_server/models/deepseek-coder-1.3b"),
        ]
        
        for path in potential_paths:
            if path.exists() and (path / "config.json").exists():
                self.model_path = path
                return
    
    def _load_from_env(self):
        """Load from environment variables"""
        env_mappings = {
            'MODEL_PATH': ('model_path', Path),
            'EMBEDDING_MODEL': ('embedding_model', str),
            'KNOWLEDGE_BASE_PATH': ('knowledge_base_path', Path),
            'EMBEDDINGS_PATH': ('embeddings_path', Path),
            'MAX_TOKENS': ('max_tokens', int),
            'TEMPERATURE': ('temperature', float),
            'TOP_P': ('top_p', float),
            'HOST': ('host', str),
            'PORT': ('port', int),
        }
        
        for env_var, (attr_name, type_func) in env_mappings.items():
            if env_var in os.environ:
                try:
                    setattr(self, attr_name, type_func(os.environ[env_var]))
                except (ValueError, TypeError):
                    pass