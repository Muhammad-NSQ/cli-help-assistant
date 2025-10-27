# CLI Help Assistant

A natural language interface for command-line tools powered by RAG (Retrieval-Augmented Generation) and local LLMs. Ask questions about CLI commands in plain English and get accurate, context-aware responses.

## 🎯 Key Features

- **Natural Language Queries**: Ask "how do I find Python files?" instead of memorizing complex syntax
- **RAG-Powered Responses**: Retrieves relevant information from a structured knowledge base before generating answers
- **Fast Response Times**: Model and embeddings stay loaded in memory (~1-2 seconds per query)
- **Extensible Knowledge Base**: Easily add new tools and commands via YAML files
- **GPU-Accelerated**: Leverages NVIDIA GPUs for fast inference
- **Docker-Ready**: Production-ready containerized deployment

## 🏗️ Architecture

The system uses a client-server architecture with RAG (Retrieval-Augmented Generation):
```
┌─────────────┐          ┌──────────────────────────┐
│             │  HTTP    │   Model Server           │
│  CLI Client │ ────────>│  • LLM (loaded in VRAM)  │
│  (thin)     │          │  • Embeddings (loaded)   │
│             │ <────────│  • FAISS index (loaded)  │
└─────────────┘          │  • Knowledge base        │
                         └──────────────────────────┘
```

**Why this architecture?**
- **Model Server**: Keeps the LLM, embeddings, and FAISS index loaded in memory for instant responses
- **CLI Client**: Lightweight HTTP client that just sends queries and displays results
- **Result**: 5-10x faster than loading everything on each query

## 🚀 Quick Start

### Prerequisites

- **GPU**: NVIDIA GPU with 8GB+ VRAM (or 14GB+ for larger models)
- **Docker & Docker Compose**: For containerized deployment
- **NVIDIA Container Toolkit**: For GPU support in Docker

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/cli-help-assistant.git
cd cli-help-assistant
```

### 2. Download the Model
```bash
# Install huggingface_hub
pip install huggingface-hub

# Download DeepSeek Coder 6.7B (recommended)
python3 << 'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="deepseek-ai/deepseek-coder-6.7b-instruct",
    local_dir="./model_server/models/deepseek-coder-6.7b-instruct",
    local_dir_use_symlinks=False
)
EOF
```

### 3. Build Knowledge Base
```bash
# Build embeddings from YAML knowledge files
docker-compose run --rm --entrypoint sh model-server -c \
  "python3 -c 'from rag_service import RAGService; from config import ModelServerConfig; \
  r = RAGService(ModelServerConfig()); r.load(); r.rebuild()'"
```

### 4. Start the Model Server
```bash
# Start model server (loads LLM + RAG into memory)
docker-compose up model-server -d

# Wait for startup (~30-60 seconds)
docker-compose logs -f model-server

# Look for: "Model loaded successfully" and "RAG service loaded successfully"
```

### 5. Test the CLI
```bash
# Ask a question
docker-compose --profile cli run --rm cli-help ask "how do I find Python files?"

# Get command explanation
docker-compose --profile cli run --rm cli-help explain "find"

# List available tools
docker-compose --profile cli run --rm cli-help list-tools
```

## 📖 Usage Examples
```bash
# General queries
cli-help ask "how do I find large files?"
cli-help ask "show me git workflow"
cli-help ask "how to search inside files?"

# Command explanations
cli-help explain "find"
cli-help explain "git status"

# Tool examples
cli-help examples git
cli-help examples core_utils

# List available tools
cli-help list-tools
```

## 🔧 Configuration

### Docker Compose Configuration

Edit `docker-compose.yml` to configure the model server:
```yaml
environment:
  - MODEL_PATH=/app/models/deepseek-coder-6.7b-instruct
  - TEMPERATURE=0.0          # Lower = more deterministic
  - MAX_TOKENS=150           # Response length
  - KNOWLEDGE_BASE_PATH=/app/knowledge
  - EMBEDDINGS_PATH=/app/knowledge/processed
```

### Available Models

| Model | VRAM | Quality | Speed |
|-------|------|---------|-------|
| deepseek-coder-1.3b | 3GB | Poor (hallucinates) | Fast |
| deepseek-coder-6.7b-instruct | 14GB | Good | Medium |
| codellama-7b-instruct | 14GB | Good | Medium |

## 📚 Adding New Tools to Knowledge Base

1. Create a YAML file in `knowledge/commands/`:
```yaml
tool_name: "mytool"
description: "What the tool does"
category: "development"

commands:
  - name: "mytool action"
    description: "What this command does"
    syntax: "mytool [options] [arguments]"
    examples:
      - command: "mytool --help"
        description: "Show help information"
      - command: "mytool run --verbose"
        description: "Run with verbose output"
    common_options:
      - flag: "--help"
        description: "Show help"
      - flag: "--verbose"
        description: "Increase verbosity"
```

2. Rebuild the knowledge base:
```bash
curl -X POST http://localhost:8000/rebuild-knowledge
```

## 🌐 API Endpoints

The model server exposes these REST endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and service status |
| `/query` | POST | Ask a question (RAG + LLM) |
| `/explain` | POST | Explain a specific command |
| `/tools` | GET | List all available tools |
| `/tools/{tool}/examples` | GET | Get examples for a tool |
| `/rebuild-knowledge` | POST | Rebuild embeddings |

### Example API Usage
```bash
# Health check
curl http://localhost:8000/health

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "how do I find Python files?", "top_k": 3}'

# Explain command
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"command": "find"}'
```

## 📁 Project Structure
```
cli-help-assistant/
├── cli_help/              # CLI client (thin HTTP client)
│   ├── main.py           # CLI entry point
│   └── ...
├── model_server/         # Model server (LLM + RAG)
│   ├── main.py          # FastAPI server
│   ├── llm_service.py   # LLM loading and inference
│   ├── rag_service.py   # RAG (embeddings + FAISS)
│   ├── config.py        # Configuration
│   └── models/          # Downloaded LLM models (not in git)
├── knowledge/            # Knowledge base
│   ├── commands/        # Tool definitions (YAML)
│   └── processed/       # Generated embeddings (not in git)
├── docker-compose.yml   # Container orchestration
├── Dockerfile           # CLI client container
├── Dockerfile.model_server  # Model server container
└── requirements.txt     # Python dependencies
```

## ⚡ Performance

### Response Times

- **Cold start**: 30-60 seconds (model + embeddings loading)
- **Warm queries**: 1-2 seconds per query
- **Memory usage**: 
  - 1.3B model: ~3GB VRAM
  - 6.7B model: ~14GB VRAM

### Why So Fast?

1. **Persistent model loading**: Model stays in VRAM between queries
2. **Cached embeddings**: FAISS index loaded once at startup
3. **Pre-computed knowledge**: Embeddings built offline, not per-query
4. **GPU acceleration**: CUDA-optimized inference

## 🐛 Troubleshooting

### Model Server Won't Start
```bash
# Check logs
docker-compose logs model-server

# Common issues:
# 1. Not enough VRAM - try smaller model
# 2. Model files missing - re-download
# 3. NVIDIA drivers - check nvidia-smi
```

### Poor Quality Responses
```bash
# Issue: Using 1.3B model (too small, hallucinates)
# Solution: Upgrade to 6.7B model

# Update MODEL_PATH in docker-compose.yml
MODEL_PATH=/app/models/deepseek-coder-6.7b-instruct
```

### CLI Can't Connect
```bash
# Check server is running
curl http://localhost:8000/health

# Check network
docker network ls | grep cli-help

# Restart services
docker-compose down
docker-compose up model-server -d
```

## 🛠️ Development

### Local Development (without Docker)
```bash
# Install dependencies
pip install -r requirements.txt

# Start model server
cd model_server
python main.py

# Test CLI (in another terminal)
export MODEL_SERVER_URL=http://localhost:8000
python -m cli_help.main ask "test query"
```

### Running Tests
```bash
# Test model server health
curl http://localhost:8000/health

# Test query endpoint
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "top_k": 3}'
```

## 📋 Requirements

### System Requirements

- **RAM**: 16GB+ system RAM
- **GPU**: NVIDIA GPU with 8GB+ VRAM (14GB+ recommended)
- **Storage**: 30GB+ (models, dependencies, docker images)
- **OS**: Linux (Ubuntu 20.04+ recommended)

### Software Requirements

- Docker & Docker Compose
- NVIDIA Container Toolkit
- Python 3.11+ (if running without Docker)

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **DeepSeek**: For the excellent Coder models
- **Sentence Transformers**: For the embedding models
- **FAISS**: For fast similarity search
- **FastAPI**: For the excellent web framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/cli-help-assistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/cli-help-assistant/discussions)

---

Made with ❤️ for the CLI community