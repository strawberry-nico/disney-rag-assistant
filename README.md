# 🏰 Disney RAG Assistant (Local-First MVP)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-v0.1-green)
![ChromaDB](https://img.shields.io/badge/Vector_DB-Chroma-orange)
![Deploy](https://img.shields.io/badge/Deploy-Local-purple)

一个基于 **RAG (Retrieval-Augmented Generation)** 架构的垂直领域知识问答系统。
本项目不仅仅是一个 Demo，更是一次关于 **本地化部署、依赖治理、数据安全与工程规范** 的最佳实践探索。

---

## 🛠️ 技术架构 (Architecture)

本项目采用 **"Local Embedding + Cloud LLM"** 的混合架构，在保障数据隐私的同时，利用云端大模型的推理能力。

| 模块 | 技术选型 | 核心考量 |
| :--- | :--- | :--- |
| **Orchestration** | LangChain | 模块化构建检索问答链 (RetrievalQA) |
| **Embedding** | BAAI/bge-m3 | **本地部署**，支持多语言高精度语义检索，无需上传数据 |
| **Vector DB** | Chroma | 轻量级嵌入式数据库，**持久化存储**于本地，无需独立服务 |
| **LLM** | DashScope (Qwen) | 通义千问 MAX 模型，提供强大的语义理解与生成能力 |
| **Interface** | Gradio | 快速构建交互式 Web UI，支持流式输出 |
| **Infra** | uv | 使用 Rust 编写的包管理器，实现**秒级环境构建**与依赖隔离 |

---

## ✨ 核心特性 (Key Features)

### 1. 🛡️ 生产级工程规范
- **环境隔离**：严格区分应用运行环境 (`venv_app`) 与数据处理环境 (`venv_etl`)，彻底解决依赖冲突问题。
- **配置安全**：遵循 **Config as Code** 原则，通过 `.env` 管理敏感凭证，配合精细化的 `.gitignore` 策略，杜绝 API Key 泄露风险。

### 2. ⚡ 高效的本地化检索
- 集成 **BGE-M3** 模型，支持对迪士尼乐园相关文档（PDF/TXT/Markdown）的语义索引。
- 实现了基于 **Cosine Similarity** 的向量检索，并针对中文语境进行了 Prompt 优化。

### 3. 🧩 鲁棒的系统设计
- 解决了 Gradio 多线程下的 LLM Client 生命周期问题，实现了 **Request-Scoped** 连接管理。
- 包含完整的数据清洗 (ETL) 脚本，支持多模态数据（图片/表格）的预处理。

---

## 📂 项目结构 (Project Structure)

```text
DISNEY-RAG/
├── models/             # 本地 Embedding 模型 (bge-m3)
├── chroma_db/          # 向量数据库持久化文件 (GitIgnored)
├── processed_texts/    # 清洗后的知识库切片
├── src/
│   ├── app.py              # Gradio 主应用程序
│   ├── build_vector_db.py  # 向量数据库构建脚本 (ETL)
│   └── parse_images.py     # 多模态数据预处理脚本
├── .env                # 环境变量配置 (GitIgnored)
├── .gitignore          # 经过生产环境验证的忽略规则
├── requirements-app.txt # 应用运行依赖 (精简版)
└── README.md           # 项目文档

🚀 快速开始 (Quick Start)
本项目推荐使用 uv 进行依赖管理，以获得极致的安装速度与环境隔离体验。

1. 克隆项目
Bash
git clone [YOUR_GITHUB_REPO_LINK]
cd disney-rag-assistant
2. 环境配置
创建并激活虚拟环境：

Bash
# Windows
uv venv venv_app
.\venv_app\Scripts\activate

# Linux/Mac
uv venv venv_app
source venv_app/bin/activate
安装依赖：

Bash
uv pip install -r requirements-app.txt
3. 配置密钥
在项目根目录创建 .env 文件，并填入你的 DashScope API Key：

Ini, TOML
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
4. 启动应用
Bash
python src/app.py
终端显示 Running on local URL: http://127.0.0.1:7860 即表示启动成功。

📝 待办事项 (To-Do)
[ ] 引入 Cross-Encoder (Rerank) 重排序模块以提升 Top-3 准确率。

[ ] 接入多模态 Embedding 实现原生“以图搜图”。

[ ] 增加 Dockerfile 实现容器化部署。