# Test-Time Tool Evolution

一个基于大语言模型的测试时工具演化系统，能够自动生成、验证和管理可执行的 Python 工具，实现工具库的动态演化。

## 📋 项目简介

本项目实现了一个智能工具演化系统，通过以下工作流程自动生成和管理工具：

1. **工具选择**：根据用户问题，从工具库中选择最相关的工具
2. **工具调用**：使用 OpenAI API 进行函数调用
3. **新工具生成**：当现有工具无法解决问题时，通过 Chain-of-Thought (CoT) 生成新的工具代码
4. **工具验证**：在 Docker 容器中验证新生成工具的语法和可执行性
5. **工具管理**：自动去重、淘汰低使用率工具，维护工具库质量

## ✨ 核心特性

- 🔍 **智能工具选择**：基于嵌入模型和重排序模型的多阶段工具检索
- 🛠️ **自动工具生成**：使用 LLM 根据问题自动生成 Python 工具函数
- ✅ **Docker 验证**：在隔离环境中验证工具代码，自动安装缺失依赖
- 🔄 **工具去重**：使用代码嵌入相似度检测，避免添加重复工具
- 📊 **工具淘汰机制**：基于使用频率的 LRU 策略，维护工具库大小
- 📝 **完整元数据**：每个工具包含描述、输入输出参数、测试示例等信息

## 🏗️ 项目结构

```
Test-Time Tool Evolution/
├── src/                          # 源代码目录
│   ├── atomic_tool.py            # 原子工具类定义（AtomicTool）
│   ├── tool_library.py           # 工具库管理器（ToolLibrary）
│   ├── tool_selector.py          # 工具选择器（嵌入+重排序）
│   ├── main_pipeline.py          # 主处理流程（main_pipeline）
│   ├── prompt_builder.py         # CoT 提示词构建器
│   ├── openai_utils.py           # OpenAI API 封装工具
│   ├── new_tool_processor.py     # 新工具处理模块（去重逻辑）
│   ├── validate_in_docker.py     # Docker 验证模块
│   └── run.py                    # 主入口脚本
├── dataset/                      # 数据集目录
│   ├── adapt_tools/              # 适配工具库
│   ├── scibench/                 # SciBench 科学计算数据集
│   ├── scieval/                  # SciEval 科学评估数据集
│   ├── scievo/                   # SciEvo 数据集
│   └── tools_distribution/       # 工具分布数据
├── README.md                     # 中文说明文档（本文件）
├── README_CH.md                  # 中文说明文档（详细版）
└── README_EN.md                  # 英文说明文档
```

## 🔧 环境要求

### 系统要求
- Python 3.8+
- Docker Desktop（用于工具验证）
- CUDA（可选，用于 GPU 加速）

### Python 依赖

主要依赖包：
- `openai` - OpenAI API 客户端
- `sentence-transformers` - 嵌入模型
- `transformers` - Hugging Face 模型库
- `torch` - PyTorch（用于嵌入计算）
- `numpy` - 数值计算

## 📦 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd "Test-Time Tool Evolution"
```

2. **安装 Python 依赖**
```bash
pip install openai sentence-transformers transformers torch numpy
```

3. **安装并启动 Docker Desktop**
   - 下载并安装 [Docker Desktop](https://www.docker.com/products/docker-desktop)
   - 确保 Docker 服务正在运行

4. **下载模型文件**
   - BGE 嵌入模型：用于工具描述嵌入
   - 重排序模型：用于工具重排序
   - Python BERT 模型（可选）：用于代码相似度检查

## 🚀 使用方法

### 1. 配置参数

编辑 `src/run.py`，设置以下参数：

```python
# 数据路径
tool_path = "path/to/your/tool_library.json"          # 工具库保存路径
all_datas_path = "path/to/your/questions.json"       # 问题数据集路径
output_file = "path/to/output.jsonl"                  # 输出结果路径

# API 配置
cot_output_api_key = "your-api-key"                   # CoT 生成 API Key
cot_output_base_url = "your-api-base-url"             # CoT 生成 API Base URL
fun_call_api_key = "your-api-key"                     # 函数调用 API Key
fun_call_base_url = "your-api-base-url"               # 函数调用 API Base URL

# 模型路径
python_bert_model_dir = "path/to/bert-model"          # Python BERT 模型（可选）
bge_emb_model_dir = "path/to/bge-model"               # BGE 嵌入模型
reranker_model_dir = "path/to/reranker-model"         # 重排序模型
```

### 2. 准备数据格式

**工具库格式** (`tool_library.json`):
```json
{
  "tool_name": {
    "name": "tool_name",
    "code": "def tool_name(...): ...",
    "description": "工具描述",
    "input_params": {"param1": "描述1", "param2": "描述2"},
    "output_params": {"result": "输出描述"},
    "example": {"input": {...}, "result": "..."},
    "hit_count": 0,
    "created_at": "2024-01-01 00:00:00"
  }
}
```

### 3. 运行主流程

```bash
cd src
python run.py
```

## 🔄 工作流程

1. **问题分解**：将主问题分解为多个子问题
2. **工具检索**：对每个子问题，使用嵌入模型检索 top-k 候选工具
3. **工具重排序**：使用重排序模型对候选工具进行精排
4. **函数调用**：调用 OpenAI API，让模型选择并调用最合适的工具
5. **CoT 生成**：对于无法用现有工具解决的子问题，生成新的工具代码
6. **Docker 验证**：在隔离的 Docker 容器中验证新工具
7. **工具入库**：将验证通过的工具添加到工具库（去重检查）
8. **结果输出**：保存处理结果和生成的工具

## 🐳 Docker 验证

系统使用 Docker 容器验证新生成的工具：

- **隔离环境**：在干净的 Python 环境中运行代码
- **自动安装依赖**：检测缺失的包并自动安装（最多重试 3 次）
- **超时保护**：默认超时 300 秒
- **详细日志**：记录 stdout、stderr 和安装的包

## ⚙️ 配置说明

### ToolLibrary 参数

- `max_size`: 工具库最大容量（默认 50）
- `save_path`: 工具库保存路径
- `embedder`: SentenceTransformer 嵌入器

### 工具选择参数

- `top_k`: 初始检索的候选工具数量（默认 3）
- `model_dir`: BGE 嵌入模型路径
- `reranker_dir`: 重排序模型路径

### 验证参数

- `docker_image`: Docker 镜像（默认 `python:3.11-slim`）
- `timeout`: 验证超时时间（默认 300 秒）
- `max_install_retry`: 最大安装重试次数（默认 3）

## 🔍 核心模块说明

### AtomicTool
原子工具类，包含工具的所有元数据：
- 名称、代码、描述
- 输入/输出参数
- 测试示例
- 使用次数、创建时间

### ToolLibrary
工具库管理器：
- 工具的增删改查
- 嵌入向量计算和更新
- 工具淘汰和持久化

### ToolSelector
工具选择器：
- 基于嵌入的语义检索
- 基于重排序模型的精排

### MainPipeline
主处理流程：
- 协调各个模块
- 处理问题并生成工具
- 验证和入库新工具

## 引用
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

