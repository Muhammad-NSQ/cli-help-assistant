# Adding New Tools to CLI Help Assistant

This guide explains how to add new command-line tools to the knowledge base.

## Tool Definition Structure

Each tool is defined in a YAML file under `knowledge/commands/`. Here's the basic structure:

```yaml
tool_name: "tool_name"
description: "Brief description of the tool"
category: "category_name"

commands:
  - name: "command_name"
    description: "What this command does"
    syntax: "command [options] [arguments]"
    examples:
      - command: "actual command"
        description: "What this example demonstrates"
    common_options:
      - flag: "--option"
        description: "What this option does"

workflows:
  - name: "Workflow name"
    description: "Description of the workflow"
    steps:
      - "step 1 command"
      - "step 2 command"

troubleshooting:
  - problem: "Error message or issue"
    solution: "How to fix it"
```

## Steps to Add a New Tool

1. Create a new YAML file: `knowledge/commands/your_tool.yaml`
2. Follow the structure above
3. Include 5-10 most common commands
4. Add practical examples for each command
5. Validate the file: `python scripts/validate_knowledge.py`
6. Rebuild embeddings: `cli-help build-knowledge`

## Best Practices

- Focus on the most commonly used commands
- Include practical, real-world examples
- Keep descriptions concise but informative
- Add troubleshooting for common errors
- Group related commands together
