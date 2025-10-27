'''Configuration management'''

import yaml
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    '''Configuration for CLI Help Assistant'''

    def __init__(self, config_path: Optional[str] = None):
        # Default values - Model settings
        self.model_name = "deepseek-coder"
        self.model_path = "./models/deepseek-coder-v2-lite"
        self.embedding_model = "all-MiniLM-L6-v2"
        
        # Knowledge base settings
        self.knowledge_base_path = Path("./knowledge")
        self.embeddings_path = Path("./knowledge/processed")
        self.chunk_size = 512
        
        # Generation parameters
        self.max_tokens = 1024
        self.temperature = 0.1
        self.top_k_results = 5
        self.top_p = 0.9
        
        # Model server settings
        self.model_server_url = "http://localhost:8000"
        self.model_server_timeout = 30
        self.model_server_retries = 3
        self.use_model_server = True  # If False, fall back to local model
        
        # Load from config file if provided
        if config_path and Path(config_path).exists():
            self._load_config(config_path)

        # Override with environment variables
        self._load_from_env()

        # Ensure paths exist
        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)
        self.embeddings_path.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str):
        '''Load configuration from YAML file'''
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        for key, value in config_data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def _load_from_env(self):
        '''Load configuration from environment variables'''
        env_mappings = {
            'MODEL_NAME': 'model_name',
            'MODEL_PATH': 'model_path',
            'EMBEDDING_MODEL': 'embedding_model',
            'KNOWLEDGE_BASE_PATH': 'knowledge_base_path',
            'EMBEDDINGS_PATH': 'embeddings_path',
            'MAX_TOKENS': 'max_tokens',
            'TEMPERATURE': 'temperature',
            'TOP_K_RESULTS': 'top_k_results',
            'TOP_P': 'top_p',
            'CHUNK_SIZE': 'chunk_size',
            
            # Model server settings
            'MODEL_SERVER_URL': 'model_server_url',
            'MODEL_SERVER_TIMEOUT': 'model_server_timeout',
            'MODEL_SERVER_RETRIES': 'model_server_retries',
            'USE_MODEL_SERVER': 'use_model_server',
        }

        for env_var, attr_name in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                # Convert to appropriate type
                if attr_name in ['max_tokens', 'top_k_results', 'chunk_size', 'model_server_timeout', 'model_server_retries']:
                    value = int(value)
                elif attr_name in ['temperature', 'top_p']:
                    value = float(value)
                elif attr_name in ['knowledge_base_path', 'embeddings_path']:
                    value = Path(value)
                elif attr_name == 'use_model_server':
                    value = value.lower() in ('true', '1', 'yes', 'on')

                setattr(self, attr_name, value)
    
    def get_model_server_config(self) -> dict:
        '''Get model server specific configuration'''
        return {
            'url': self.model_server_url,
            'timeout': self.model_server_timeout,
            'retries': self.model_server_retries,
            'enabled': self.use_model_server
        }
    
    def __repr__(self):
        return f"""Config(
    model_name='{self.model_name}',
    model_path='{self.model_path}',
    model_server_url='{self.model_server_url}',
    use_model_server={self.use_model_server},
    max_tokens={self.max_tokens},
    temperature={self.temperature}
)"""