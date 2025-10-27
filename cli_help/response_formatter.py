'''Format responses for better readability'''

from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.panel import Panel
from typing import List, Dict, Any

class ResponseFormatter:
    '''Format LLM responses with rich formatting'''

    def __init__(self):
        self.console = Console()

    def format_response(self, response: str, context_info: List[Dict[str, Any]]) -> str:
        '''Format a general response with context'''
        # Clean up the response
        response = response.strip()

        # Add related commands if available
        related_commands = self._extract_related_commands(context_info)
        if related_commands:
            response += "\n\n**Related commands:**\n"
            for cmd in related_commands[:3]:  # Limit to 3
                response += f"- `{cmd}`\n"

        return response

    def format_explanation(self, response: str, context_info: List[Dict[str, Any]]) -> str:
        '''Format a command explanation'''
        response = response.strip()

        # Add examples if available
        examples = self._extract_examples(context_info)
        if examples:
            response += "\n\n**Examples:**\n"
            for example in examples[:2]:  # Limit to 2
                response += f"```bash\n{example['command']}\n```\n{example['description']}\n\n"

        return response

    def _extract_related_commands(self, context_info: List[Dict[str, Any]]) -> List[str]:
        '''Extract related commands from context'''
        commands = []
        for info in context_info:
            if info.get('type') == 'command':
                commands.append(info.get('command', ''))
        return list(set(filter(None, commands)))

    def _extract_examples(self, context_info: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        '''Extract examples from context'''
        examples = []
        for info in context_info:
            if info.get('type') == 'example' and 'example' in info:
                examples.append(info['example'])
        return examples
