'''Build and manage the knowledge base'''

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

logger = logging.getLogger(__name__)

class KnowledgeBuilder:
    '''Build and search knowledge base with embeddings'''

    def __init__(self, config):
        self.config = config
        self.embedding_model = None
        self.index = None
        self.text_chunks = []
        self.metadata = []
        self._load_embedding_model()

    def _load_embedding_model(self):
        '''Load sentence transformer model for embeddings'''
        try:
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            logger.info(f"Loaded embedding model: {self.config.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")

    def load_knowledge_base(self) -> Dict[str, Any]:
        '''Load all YAML knowledge files'''
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

    def build_embeddings(self):
        '''Build embeddings for all knowledge base content'''
        if not self.embedding_model:
            raise Exception("Embedding model not loaded")

        knowledge_data = self.load_knowledge_base()

        # Extract text chunks for embedding
        self.text_chunks = []
        self.metadata = []

        for tool_name, tool_data in knowledge_data.items():
            # Add tool description
            if 'description' in tool_data:
                self.text_chunks.append(f"{tool_name}: {tool_data['description']}")
                self.metadata.append({'type': 'tool_description', 'tool': tool_name})

            # Add commands
            for command in tool_data.get('commands', []):
                # Command description
                text = f"{command['name']}: {command.get('description', '')}"
                self.text_chunks.append(text)
                self.metadata.append({'type': 'command', 'tool': tool_name, 'command': command['name']})

                # Examples
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
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        # Save embeddings and metadata
        self._save_embeddings(embeddings)

        logger.info(f"Built embeddings index with {len(self.text_chunks)} entries")

    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        '''Search for similar content'''
        if not self.index or not self.embedding_model:
            logger.warning("Embeddings not loaded, falling back to keyword search")
            return self._keyword_search(query, top_k)

        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)

            # Search
            scores, indices = self.index.search(query_embedding, top_k)

            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    result['text'] = self.text_chunks[idx]
                    result['score'] = float(score)
                    results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return self._keyword_search(query, top_k)

    def search_command(self, command: str) -> List[Dict[str, Any]]:
        '''Search for specific command'''
        # Try exact match first
        for i, metadata in enumerate(self.metadata):
            if metadata.get('type') == 'command' and command.lower() in metadata.get('command', '').lower():
                return [{'text': self.text_chunks[i], **metadata}]

        # Fall back to similarity search
        return self.search_similar(command, top_k=3)

    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        '''Get information for specific tool'''
        knowledge_data = self.load_knowledge_base()
        return knowledge_data.get(tool_name, {})

    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        '''Simple keyword-based search fallback'''
        query_words = query.lower().split()
        results = []

        for i, text in enumerate(self.text_chunks):
            score = sum(1 for word in query_words if word in text.lower())
            if score > 0:
                result = self.metadata[i].copy()
                result['text'] = text
                result['score'] = score
                results.append(result)

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def _save_embeddings(self, embeddings):
        '''Save embeddings and metadata to disk'''
        embeddings_dir = self.config.embeddings_path
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(embeddings_dir / "embeddings.index"))

        # Save metadata
        with open(embeddings_dir / "metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2)

        # Save text chunks
        with open(embeddings_dir / "text_chunks.json", 'w') as f:
            json.dump(self.text_chunks, f, indent=2)

        logger.info(f"Saved embeddings to {embeddings_dir}")

    def load_embeddings(self):
        '''Load embeddings from disk'''
        embeddings_dir = self.config.embeddings_path

        try:
            # Load FAISS index
            self.index = faiss.read_index(str(embeddings_dir / "embeddings.index"))

            # Load metadata
            with open(embeddings_dir / "metadata.json", 'r') as f:
                self.metadata = json.load(f)

            # Load text chunks
            with open(embeddings_dir / "text_chunks.json", 'r') as f:
                self.text_chunks = json.load(f)

            logger.info(f"Loaded embeddings from {embeddings_dir}")

        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise
