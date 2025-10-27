"""
RAG Service - Handles knowledge base, embeddings, and semantic search
"""

import logging
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

logger = logging.getLogger(__name__)

class RAGService:
    """Service for managing RAG (Retrieval-Augmented Generation)"""
    
    def __init__(self, config):
        self.config = config
        self.embedding_model = None
        self.index = None
        self.text_chunks = []
        self.metadata = []
        self.knowledge_data = {}
        self._loaded = False
        
    def load(self):
        """Load embeddings and knowledge base"""
        try:
            # Load embedding model
            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            
            # Load knowledge base
            logger.info("Loading knowledge base...")
            self.knowledge_data = self._load_knowledge_base()
            
            # Try to load pre-built embeddings
            try:
                self._load_embeddings()
                logger.info(f"Loaded {len(self.text_chunks)} embeddings from disk")
            except Exception as e:
                logger.warning(f"Could not load embeddings: {e}")
                logger.info("Building embeddings from scratch...")
                self._build_embeddings()
            
            self._loaded = True
            logger.info("RAG service loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load RAG service: {e}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if RAG service is loaded"""
        return self._loaded and self.embedding_model is not None and self.index is not None
    
    def query(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Search for relevant information"""
        if not self.is_loaded():
            raise RuntimeError("RAG service not loaded")
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, top_k)
            
            relevant_info = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.metadata) and score > 0.4:  # Relevance threshold
                    result = self.metadata[idx].copy()
                    result['text'] = self.text_chunks[idx]
                    result['score'] = float(score)
                    relevant_info.append(result)
            
            # Extract related commands
            related_commands = list(set([
                info.get('command', '') 
                for info in relevant_info 
                if info.get('type') == 'command' and info.get('command')
            ]))
            
            return {
                'relevant_info': relevant_info,
                'related_commands': related_commands
            }
            
        except Exception as e:
            logger.error(f"Query error: {e}")
            return {'relevant_info': [], 'related_commands': []}
    
    def search_command(self, command: str) -> Dict[str, Any]:
        """Search for specific command"""
        # Try exact match first
        for i, metadata in enumerate(self.metadata):
            if metadata.get('type') == 'command' and command.lower() in metadata.get('command', '').lower():
                return {
                    'relevant_info': [{'text': self.text_chunks[i], **metadata}],
                    'related_commands': [command]
                }
        
        # Fall back to similarity search
        return self.query(command, top_k=3)
    
    def list_tools(self) -> List[str]:
        """List all available tools"""
        return list(self.knowledge_data.keys())
    
    def get_examples(self, tool: str) -> List[str]:
        """Get examples for a specific tool"""
        tool_info = self.knowledge_data.get(tool, {})
        
        if not tool_info:
            return [f"No information found for '{tool}'"]
        
        examples = []
        for command in tool_info.get('commands', []):
            for example in command.get('examples', []):
                examples.append(
                    f"$ {example['command']}\n  → {example['description']}"
                )
        
        return examples if examples else [f"No examples found for '{tool}'"]
    
    def clean_response(self, response: str) -> str:
        """Clean up model output"""
        if not response:
            return response
        
        # Stop at common rambling indicators
        cutoff_phrases = [
            "\n\nRemember", "\n\nPlease", "\n\nNote:", "\n\nAlso note",
            "\n\nHowever", "\n\nIn case", "\n\nFinally", "\n\nLastly",
            "\n3)", "\n4)", "\n5)"
        ]
        
        for phrase in cutoff_phrases:
            if phrase in response:
                response = response.split(phrase)[0]
        
        # Keep only first 2-3 paragraphs
        paragraphs = [p.strip() for p in response.split('\n\n') if len(p.strip()) > 20]
        if paragraphs:
            response = '\n\n'.join(paragraphs[:3])
        
        return response.strip()
    
    def rebuild(self):
        """Rebuild embeddings from knowledge base"""
        logger.info("Rebuilding knowledge base...")
        self._build_embeddings()
        self._save_embeddings()
        logger.info("Knowledge base rebuilt successfully")
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load all YAML knowledge files"""
        knowledge_data = {}
        commands_dir = self.config.knowledge_base_path / "commands"
        
        if not commands_dir.exists():
            logger.warning(f"Commands directory not found: {commands_dir}")
            return knowledge_data
        
        for yaml_file in commands_dir.glob("*.yaml"):
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    if data and 'tool_name' in data:
                        knowledge_data[data['tool_name']] = data
                        logger.debug(f"Loaded knowledge for {data['tool_name']}")
            except Exception as e:
                logger.error(f"Error loading {yaml_file}: {e}")
        
        return knowledge_data
    
    def _build_embeddings(self):
        """Build embeddings from knowledge base"""
        self.text_chunks = []
        self.metadata = []
        
        for tool_name, tool_data in self.knowledge_data.items():
            # Add tool description
            if 'description' in tool_data:
                self.text_chunks.append(f"{tool_name}: {tool_data['description']}")
                self.metadata.append({'type': 'tool_description', 'tool': tool_name})
            
            # Add commands
            for command in tool_data.get('commands', []):
                text = f"{command['name']}: {command.get('description', '')}"
                self.text_chunks.append(text)
                self.metadata.append({
                    'type': 'command',
                    'tool': tool_name,
                    'command': command['name']
                })
                
                # Add examples
                for example in command.get('examples', []):
                    text = f"{example['command']}: {example['description']}"
                    self.text_chunks.append(text)
                    self.metadata.append({
                        'type': 'example',
                        'tool': tool_name,
                        'command': command['name'],
                        'example': example
                    })
        
        logger.info(f"Extracted {len(self.text_chunks)} text chunks")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(self.text_chunks)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        logger.info(f"Built FAISS index with {len(self.text_chunks)} entries")
    
    def _save_embeddings(self):
        """Save embeddings to disk"""
        embeddings_dir = self.config.embeddings_path
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(embeddings_dir / "embeddings.index"))
        
        with open(embeddings_dir / "metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        with open(embeddings_dir / "text_chunks.json", 'w') as f:
            json.dump(self.text_chunks, f, indent=2)
        
        logger.info(f"Saved embeddings to {embeddings_dir}")
    
    def _load_embeddings(self):
        """Load embeddings from disk"""
        embeddings_dir = self.config.embeddings_path
        
        self.index = faiss.read_index(str(embeddings_dir / "embeddings.index"))
        
        with open(embeddings_dir / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        with open(embeddings_dir / "text_chunks.json", 'r') as f:
            self.text_chunks = json.load(f)