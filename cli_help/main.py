#!/usr/bin/env python3
'''
CLI Help Assistant - Thin client
'''

import click
import requests
from rich.console import Console
from rich.markdown import Markdown

console = Console()

SERVER_URL = "http://model-server:8000"

@click.group()
def cli():
    '''CLI Help Assistant - Natural language help for command-line tools'''
    pass

@cli.command()
@click.argument('question')
def ask(question):
    '''Ask a question about CLI commands'''
    try:
        response = requests.post(
            f"{SERVER_URL}/query",
            json={"query": question, "top_k": 3},
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        console.print(data['answer'])
        
        if data.get('sources'):
            console.print(f"\n[dim]Related: {', '.join([f'`{s}`' for s in data['sources']])}[/dim]")
            
    except requests.exceptions.ConnectionError:
        console.print("[red]Error: Cannot connect to model server[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

@cli.command()
@click.argument('command')
def explain(command):
    '''Get detailed explanation of a specific command'''
    try:
        response = requests.post(
            f"{SERVER_URL}/explain",
            json={"command": command},
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        console.print(data['answer'])
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

@cli.command()
@click.argument('tool')
def examples(tool):
    '''Show examples for a specific tool'''
    try:
        response = requests.get(
            f"{SERVER_URL}/tools/{tool}/examples",
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        for example in data['examples']:
            console.print(example)
            console.print()
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

@cli.command()
def list_tools():
    '''List all available tools'''
    try:
        response = requests.get(f"{SERVER_URL}/tools", timeout=10)
        response.raise_for_status()
        
        data = response.json()
        console.print("Available tools:")
        for tool in data['tools']:
            console.print(f"  • {tool}")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

@cli.command()
def build_knowledge():
    '''Rebuild the knowledge base (on server)'''
    try:
        console.print("Rebuilding knowledge base on server...")
        response = requests.post(f"{SERVER_URL}/rebuild-knowledge", timeout=120)
        response.raise_for_status()
        
        console.print("[green]Knowledge base rebuilt successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

if __name__ == '__main__':
    cli()