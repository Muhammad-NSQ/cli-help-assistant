# CLI Help Assistant Usage Examples

## Basic Usage

```bash
# Ask natural language questions
cli-help ask "how do I find large files?"
cli-help ask "what does git stash do?"
cli-help ask "show me grep examples"

# Get detailed command explanations
cli-help explain "find"
cli-help explain "git commit"

# See examples for specific tools
cli-help examples "git"
cli-help examples "find"

# List all available tools
cli-help list-tools
```

## Docker Usage

```bash
# Start the container
docker-compose up -d

# Run commands in container
docker-compose exec cli-help cli-help ask "how do I search text in files?"

# Build knowledge base in container
docker-compose exec cli-help cli-help build-knowledge
```

## Advanced Queries

The assistant can handle various types of questions:

- **How-to questions**: "how do I find Python files modified today?"
- **Explanation requests**: "what does the -la flag do in ls?"
- **Alternative methods**: "different ways to search for text"
- **Troubleshooting**: "why am I getting permission denied?"
