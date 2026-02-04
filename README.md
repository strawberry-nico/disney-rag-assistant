# Disney Knowledge Assistant

一个基于 RAG 的本地化知识问答系统，支持 PDF/DOCX/PPTX 文档智能解析与问答。

## ✨ 特性
- 📄 自动解析多格式文档（含图片 OCR）
- 🧠 使用 BGE-M3 本地 embedding 模型（中文优化）
- 🔍 基于 Chroma 的高效向量检索
- 💬 Gradio 交互界面，支持自然语言提问

## 🚀 快速开始
安装依赖：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
