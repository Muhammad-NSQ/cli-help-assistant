'''Handle user queries and retrieve relevant information'''

import logging
from typing import List, Dict, Any
from pathlib import Path

from .llm_interface import LLMInterface
from .knowledge_builder import KnowledgeBuilder
from .response_formatter import ResponseFormatter

logger = logging.getLogger(__name__)

class QueryHandler:
    '''Process user queries and generate responses'''

    def __init__(self, config):
        self.config = config
        self.knowledge_builder = KnowledgeBuilder(config)
        self.formatter = ResponseFormatter()

        # Create LLM interface
        logger.info("Connecting to model server...")
        self.llm = LLMInterface(config)

        # Load embeddings from disk
        try:
            self.knowledge_builder.load_embeddings()
            logger.info(f"Loaded {len(self.knowledge_builder.text_chunks)} embeddings successfully")
        except Exception as e:
            logger.warning(f"Could not load embeddings: {e}")

        # Load knowledge base
        try:
            self.knowledge_data = self.knowledge_builder.load_knowledge_base()
            logger.info(f"Loaded knowledge for {len(self.knowledge_data)} tools")
        except Exception as e:
            logger.warning(f"Could not load knowledge base: {e}")
            self.knowledge_data = {}

    def process_query(self, query: str) -> str:
        '''Process a natural language query'''
        try:
            # Retrieve relevant information
            relevant_info = self.knowledge_builder.search_similar(query, top_k=self.config.top_k_results)
            
            # Filter out low-relevance results
            relevant_info = [info for info in relevant_info if info.get('score', 0) > 0.3]
            
            if not relevant_info:
                return f"Sorry, I don't have specific information about '{query}'. Try 'cli-help list-tools' to see available commands."

            # Format context for LLM
            context = self._format_context_strict(relevant_info)

            # Generate response using LLM
            prompt = self._build_strict_prompt(query, context)
            response = self.llm.generate_response(prompt)

            # Format the final response
            formatted_response = self.formatter.format_response(response, relevant_info)

            return formatted_response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Sorry, I encountered an error: {e}"

    def get_command_explanation(self, command: str) -> str:
        '''Get explanation for a specific command'''
        try:
            # Search for exact command match
            relevant_info = self.knowledge_builder.search_command(command)

            if not relevant_info:
                return f"Sorry, I don't have information about '{command}'. Try 'cli-help list-tools' to see available commands."

            context = self._format_context_strict(relevant_info)
            prompt = self._build_explanation_prompt(command, context)

            response = self.llm.generate_response(prompt)
            return self.formatter.format_explanation(response, relevant_info)

        except Exception as e:
            logger.error(f"Error explaining command: {e}")
            return f"Sorry, I encountered an error: {e}"

    def get_examples(self, tool: str) -> List[str]:
        '''Get examples for a specific tool'''
        try:
            tool_info = self.knowledge_builder.get_tool_info(tool)

            if not tool_info:
                return [f"Sorry, I don't have information about '{tool}'. Try 'cli-help list-tools' to see available tools."]

            examples = []
            for command in tool_info.get('commands', []):
                for example in command.get('examples', []):
                    examples.append(
                        f"$ {example['command']}\n  → {example['description']}"
                    )

            return examples if examples else [f"No examples found for '{tool}'"]

        except Exception as e:
            logger.error(f"Error getting examples: {e}")
            return [f"Sorry, I encountered an error: {e}"]

    def list_available_tools(self) -> List[str]:
        '''List all available tools'''
        try:
            return list(self.knowledge_data.keys())
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return []

    def _format_context_strict(self, relevant_info: List[Dict[str, Any]]) -> str:
        '''Format retrieved information as context'''
        if not relevant_info:
            return "No relevant information found."
            
        context_parts = []
        
        # Group by tool
        tools_info = {}
        for info in relevant_info:
            tool = info.get('tool', 'unknown')
            if tool not in tools_info:
                tools_info[tool] = []
            tools_info[tool].append(info)
        
        for tool, infos in tools_info.items():
            for info in infos:
                text = info.get('text', '')
                if text:
                    context_parts.append(text)
        
        return "\n".join(context_parts)

    def _build_strict_prompt(self, query: str, context: str) -> str:
        '''Build prompt with DeepSeek instruction format'''
        return f"""### Instruction:
You are a helpful assistant for command-line tools. Answer the user's question concisely using only the information provided below.

Context:
{context}

Question: {query}

### Response:
"""

    def _build_explanation_prompt(self, command: str, context: str) -> str:
        '''Build explanation prompt with DeepSeek format'''
        return f"""### Instruction:
Explain the '{command}' command concisely using the information below.

Context:
{context}

### Response:
"""

    def _format_context(self, relevant_info: List[Dict[str, Any]]) -> str:
        '''Legacy format method for compatibility'''
        return self._format_context_strict(relevant_info)

    def _build_prompt(self, query: str, context: str) -> str:
        '''Legacy prompt method for compatibility'''
        return self._build_strict_prompt(query, context)