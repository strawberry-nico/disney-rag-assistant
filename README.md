#  🏰Disney RAG Intelligent Assistant (迪士尼智能导游助手)

> 基于 LangChain + Rerank + Qwen 大模型的垂直领域知识库应用。
> 专注于上海迪士尼乐园游玩攻略，提供“米奇”人设的沉浸式问答体验。

## 📖 项目简介 (Introduction)

这是一个工业级标准的 RAG (Retrieval-Augmented Generation) 演示项目。旨在解决通用大模型在特定垂直领域（迪士尼乐园）知识幻觉的问题。

本项目采用了 **工程化环境隔离 (Environment Isolation)** 设计模式，将 **ETL 数据清洗**（依赖重、冲突多）与 **App 应用服务**（追求稳定、轻量）的运行环境严格拆分，彻底解决了传统 RAG 项目中“文档解析库”与“推理库”版本打架的问题。

## ✨ 核心亮点 (Key Features)

* **🛡️ 双环境工程架构**:
* **ETL Env**: 专用于 PDF 解析、OCR 识别、脏数据清洗（依赖 `pdfplumber`, `unstructured` 等）。
* **App Env**: 专用于 Gradio 界面渲染、Rerank 推理、大模型交互（依赖 `gradio`, `torch` 等）。


* **🚀 双阶段检索 (Two-Stage Retrieval)**:
* **Recall**: 使用 `BAAI/bge-m3` 进行大规模向量召回。
* **Rerank**: 引入 `BAAI/bge-reranker-v2-m3` Cross-Encoder 模型（GPU 加速），对检索结果进行语义精排。


* **🧠 查询重写 (Query Rewrite)**: 利用 LLM 对用户口语化问题进行关键词扩展，提升长尾问题召回率。
* **🔄 数据闭环 (Data Flywheel)**: 内置 RLHF 反馈机制，结构化存储用户点赞/点踩数据 (`jsonl`)，为后续模型微调积累数据。

## 🏗️ 系统架构 (Architecture)

```mermaid
graph TD
    User[用户提问] --> Rewrite[Query Rewrite / 查询重写]
    Rewrite --> Search[Vector Search / 向量召回]

    subgraph "Knowledge Base / 知识库 (ETL Env)"
        Docs[迪士尼文档/图片] --> ETL[ETL清洗 & OCR]
        ETL --> ChromaDB[(Chroma 向量库)]
    end

    ChromaDB -.-> Search
    Search --> Candidates[Top-50 候选文档]

    Candidates --> Rerank[Cross-Encoder Rerank / 重排序]
    Rerank -- GPU加速 --> TopK[Top-3 高信度文档]

    TopK --> Context[Context 组装]
    Context --> LLM[Qwen-Max 大模型]
    LLM --> Answer[米奇回答]

    Answer --> Feedback[用户反馈 (👍/👎)]
    Feedback --> Log[(user_feedback.jsonl)]

```

## 🛠️ 技术栈 (Tech Stack)

* **LLM**: 通义千问 Qwen-max (via DashScope API)
* **Embedding**: BAAI/bge-m3
* **Rerank**: BAAI/bge-reranker-v2-m3
* **Vector DB**: ChromaDB
* **UI/UX**: Gradio 4.35 (Custom Theme)
* **Env Manager**: uv / Conda (Dual Environment Strategy)

## 🚀 快速开始 (Quick Start)

本项目严格遵循**环境隔离**原则，请务必分别创建两个虚拟环境。

### 1. 克隆项目

```bash
git clone https://github.com/your-username/disney-rag-assistant.git
cd disney-rag-assistant

```

### 2. 环境构建 (双环境隔离)

#### 🅰️ 构建 App 运行环境 (App Runtime)

*用途：启动 Web 界面、运行 RAG 推理、API 调用。*

```bash
# 1. 创建名为 .venv 的虚拟环境
uv venv .venv

# 2. 激活环境
source .venv/bin/activate

# 3. 安装应用侧依赖 (轻量级，版本锁死)
uv pip install -r requirements-app.txt

# 4. 退出环境
deactivate

```

#### 🅱️ 构建 ETL 清洗环境 (ETL Runtime)

*用途：解析 PDF、OCR 识别、构建向量数据库。*

```bash
# 1. 创建名为 .venv-etl 的虚拟环境
uv venv .venv-etl

# 2. 激活环境
source .venv-etl/bin/activate

# 3. 安装清洗侧依赖 (包含 OCR 等重型库)
uv pip install -r requirements-etl.txt

# 4. 退出环境
deactivate

```

### 3. 配置 API Key

请确保拥有阿里云 DashScope 的 API Key。

```bash
export DASHSCOPE_API_KEY="sk-你的密钥"

```

### 4. 运行步骤 (按需切换环境)

#### 步骤一：文档解析 (使用 ETL 环境)
*⚠️ 警告：必须在 `.venv-etl` 环境下运行，用于将 PDF/图片清洗为 txt。*
```bash
# 1. 激活 ETL 环境
source .venv-etl/bin/activate

# 2. 执行解析脚本 (PDF/OCR -> txt)
python src/parse_docs.py

# 3. 运行完毕后退出
deactivate

####步骤二：构建知识库 (使用 App 环境)
⚠️ 注意：建库依赖 Embedding 模型，需切换到 App 主环境。

Bash
# 1. 激活 App 环境
source .venv/bin/activate

# 2. 执行建库脚本 (Embedding -> ChromaDB)
python src/build_vector_db.py

#### 步骤三：启动应用 (使用 App 环境)
Bash
# (保持在 App 环境中)
# 启动 Web 服务
python src/app.py

启动成功后，访问终端显示的链接（AutoDL 用户请使用“自定义服务”访问端口 6006）。

## 📂 目录结构

```text
disney-rag-assistant/
├── src/
│   ├── app.py                 # [App环境] 主应用程序
│   ├── build_vector_db.py     # [ETL环境] 离线建库脚本
│   └── parse_docs.py          # [ETL环境] 文档解析模块
├── chroma_db/                 # 持久化向量数据库
├── .venv/                     # [App Runtime] 隔离的应用运行环境
├── .venv-etl/                 # [ETL Runtime] 隔离的数据清洗环境
├── requirements-app.txt       # App 依赖清单 (Gradio, Rerank)
├── requirements-etl.txt       # ETL 依赖清单 (OCR, PDFPlumber)
├── user_feedback.jsonl        # 用户反馈数据日志
└── README.md                  # 项目说明文档

```

## 🔮 未来规划 (Roadmap)

* [ ] **容器化部署**: 增加 Docker 支持，实现云端弹性伸缩。
* [ ] **混合检索 (Hybrid Search)**: 引入 ElasticSearch，增加关键词 (BM25) 检索链路。
* [ ] **多轮对话**: 增加 Session History 管理，支持追问。
* [ ] **数据库迁移**: 将 `jsonl` 反馈数据迁移至 PostgreSQL/MySQL 以支持高并发。

---

*Created by [strawberry-nico] | Powered by AutoDL & LangChain*

```
