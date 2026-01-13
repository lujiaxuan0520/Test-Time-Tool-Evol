# Test-Time Tool Evolution

A test-time tool evolution system based on large language models that automatically generates, validates, and manages executable Python tools, enabling dynamic evolution of the tool library.

## 📋 Project Overview

This project implements an intelligent tool evolution system that automatically generates and manages tools through the following workflow:

1. **Tool Selection**: Select the most relevant tools from the tool library based on user questions
2. **Tool Invocation**: Use OpenAI API for function calling
3. **New Tool Generation**: Generate new tool code via Chain-of-Thought (CoT) when existing tools cannot solve the problem
4. **Tool Validation**: Validate the syntax and executability of newly generated tools in Docker containers
5. **Tool Management**: Automatically deduplicate and evict low-usage tools to maintain tool library quality

## ✨ Key Features

- 🔍 **Intelligent Tool Selection**: Multi-stage tool retrieval based on embedding models and reranking models
- 🛠️ **Automatic Tool Generation**: Use LLM to automatically generate Python tool functions based on questions
- ✅ **Docker Validation**: Validate tool code in isolated environments with automatic dependency installation
- 🔄 **Tool Deduplication**: Use code embedding similarity detection to avoid adding duplicate tools
- 📊 **Tool Eviction Mechanism**: LRU strategy based on usage frequency to maintain tool library size
- 📝 **Complete Metadata**: Each tool includes description, input/output parameters, test examples, and more

## 🏗️ Project Structure

```
Test-Time Tool Evolution/
├── src/                          # Source code
│   ├── atomic_tool.py            # Atomic tool definition (AtomicTool)
│   ├── tool_library.py           # Tool library manager (ToolLibrary)
│   ├── tool_selector.py          # Tool selector (embedding + reranking)
│   ├── main_pipeline.py          # Main pipeline (main_pipeline)
│   ├── prompt_builder.py         # CoT prompt builder
│   ├── openai_utils.py           # OpenAI API wrapper
│   ├── new_tool_processor.py     # New tool processing module (dedup logic)
│   ├── validate_in_docker.py     # Docker validation module
│   └── run.py                    # Main entry script
├── dataset/                      # Dataset directory
│   ├── adapt_tools/              # Adapted tool library
│   ├── scibench/                 # SciBench scientific computing dataset
│   ├── scieval/                  # SciEval scientific evaluation dataset
│   ├── scievo/                   # SciEvo dataset
│   └── tools_distribution/       # Tool distribution data
├── README_CH.md                  # Chinese documentation
└── README_EN.md                  # English documentation
```

## 🔧 Requirements

### System Requirements
- Python 3.8+
- Docker Desktop (for tool validation)
- CUDA (optional, for GPU acceleration)

### Python Dependencies

Main dependencies:
- `openai` - OpenAI API client
- `sentence-transformers` - Embedding models
- `transformers` - Hugging Face model library
- `torch` - PyTorch (for embedding computation)
- `numpy` - Numerical computation

## 📦 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd "Test-Time Tool Evolution"
```

2. **Install Python dependencies**
```bash
pip install openai sentence-transformers transformers torch numpy
```

3. **Install and start Docker Desktop**
   - Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop)
   - Ensure Docker service is running

4. **Download model files**
   - BGE embedding model: For tool description embeddings
   - Reranker model: For tool reranking
   - Python BERT model (optional): For code similarity checking

## 🚀 Usage

### 1. Configure Parameters

Edit `src/run.py` and set the following parameters:

```python
# Data paths
tool_path = "path/to/your/tool_library.json"          # Tool library save path
all_datas_path = "path/to/your/questions.json"       # Question dataset path
output_file = "path/to/output.jsonl"                  # Output result path

# API configuration
cot_output_api_key = "your-api-key"                   # CoT generation API Key
cot_output_base_url = "your-api-base-url"             # CoT generation API Base URL
fun_call_api_key = "your-api-key"                     # Function calling API Key
fun_call_base_url = "your-api-base-url"               # Function calling API Base URL

# Model paths
python_bert_model_dir = "path/to/bert-model"          # Python BERT model (optional)
bge_emb_model_dir = "path/to/bge-model"               # BGE embedding model
reranker_model_dir = "path/to/reranker-model"         # Reranker model
```

### 2. Prepare Data Format

**Tool Library Format** (`tool_library.json`):
```json
{
  "tool_name": {
    "name": "tool_name",
    "code": "def tool_name(...): ...",
    "description": "Tool description",
    "input_params": {"param1": "description1", "param2": "description2"},
    "output_params": {"result": "output description"},
    "example": {"input": {...}, "result": "..."},
    "hit_count": 0,
    "created_at": "2024-01-01 00:00:00"
  }
}
```

### 3. Run Main Pipeline

```bash
cd src
python run.py
```

## 🔄 Workflow

1. **Question Decomposition**: Decompose the main question into multiple sub-questions
2. **Tool Retrieval**: For each sub-question, use embedding models to retrieve top-k candidate tools
3. **Tool Reranking**: Use reranking models to refine the ranking of candidate tools
4. **Function Calling**: Call OpenAI API to let the model select and invoke the most appropriate tool
5. **CoT Generation**: Generate new tool code for sub-questions that cannot be solved with existing tools
6. **Docker Validation**: Validate new tools in isolated Docker containers
7. **Tool Integration**: Add validated tools to the tool library (with deduplication check)
8. **Result Output**: Save processing results and generated tools

## 🐳 Docker Validation

The system uses Docker containers to validate newly generated tools:

- **Isolated Environment**: Run code in a clean Python environment
- **Automatic Dependency Installation**: Detect and automatically install missing packages (up to 3 retries)
- **Timeout Protection**: Default timeout of 300 seconds
- **Detailed Logging**: Record stdout, stderr, and installed packages

## ⚙️ Configuration

### ToolLibrary Parameters

- `max_size`: Maximum tool library capacity (default: 50)
- `save_path`: Tool library save path
- `embedder`: SentenceTransformer embedder

### Tool Selection Parameters

- `top_k`: Number of candidate tools for initial retrieval (default: 3)
- `model_dir`: BGE embedding model path
- `reranker_dir`: Reranker model path

### Validation Parameters

- `docker_image`: Docker image (default: `python:3.11-slim`)
- `timeout`: Validation timeout (default: 300 seconds)
- `max_install_retry`: Maximum installation retry count (default: 3)

## 🔍 Core Modules

### AtomicTool
Atomic tool class containing all tool metadata:
- Name, code, description
- Input/output parameters
- Test examples
- Usage count, creation time

### ToolLibrary
Tool library manager:
- Tool CRUD operations
- Embedding vector computation and updates
- Tool eviction and persistence

### ToolSelector
Tool selector:
- Embedding-based semantic retrieval
- Reranking model-based refinement

### MainPipeline
Main processing pipeline:
- Coordinates all modules
- Processes questions and generates tools
- Validates and integrates new tools

## Citation
```bibtex
@misc{lu2026statictoolstesttimetool,
      title={Beyond Static Tools: Test-Time Tool Evolution for Scientific Reasoning}, 
      author={Jiaxuan Lu and Ziyu Kong and Yemin Wang and Rong Fu and Haiyuan Wan and Cheng Yang and Wenjie Lou and Haoran Sun and Lilong Wang and Yankai Jiang and Xiaosong Wang and Xiao Sun and Dongzhan Zhou},
      year={2026},
      eprint={2601.07641},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.07641}, 
}
```
