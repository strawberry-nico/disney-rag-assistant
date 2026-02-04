import os
import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # 必须用这个新库
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatTongyi

# 1. 检查 API Key
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    # 你也可以在这里临时写死测试: DASHSCOPE_API_KEY = "sk-..."
    print("⚠️ 警告: 未检测到环境变量 DASHSCOPE_API_KEY")

# 2. 全局加载 Embedding 模型 (只加载一次，节省时间)
print("🧠 正在加载 Embedding 模型...")
try:
    embedding = HuggingFaceEmbeddings(
        model_name="./models/bge-m3",  # 指向本地模型路径
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    print("尝试使用 BAAI/bge-m3 在线模式...")
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

# 3. 全局加载向量库
if not os.path.exists("chroma_db"):
    raise FileNotFoundError("❌ 未找到 chroma_db 文件夹！请先运行 build_vector_db.py")

print("💾 正在连接向量数据库...")
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

# 4. 全局定义检索器 (关键！之前报错就是因为缺了这个)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. 定义问答函数
def rag_answer(query):
    try:
        # ✅ 关键修正：每次提问时才初始化 LLM
        # 这能解决 "client has been closed" 的网络报错
        llm = ChatTongyi(
            model="qwen-max",
            api_key=DASHSCOPE_API_KEY,
            temperature=0.3
        )

        # 创建问答链
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever, # 这里调用全局定义的 retriever
            return_source_documents=True
        )
        
        # 执行查询
        print(f"🔍 用户提问: {query}")
        result = qa_chain.invoke({"query": query})
        answer = result["result"]
        
        # 整理来源文档
        seen_sources = set()
        sources_list = []
        if "source_documents" in result:
            for doc in result["source_documents"]:
                src = os.path.basename(doc.metadata.get('source', '未知文档'))
                if src not in seen_sources:
                    sources_list.append(f"- {src}")
                    seen_sources.add(src)
        
        sources_str = "\n".join(sources_list) if sources_list else "无具体参考文档"
        return answer, sources_str

    except Exception as e:
        # 打印详细错误方便调试
        import traceback
        traceback.print_exc()
        return f"❌ 发生错误: {str(e)}", ""

# 6. 启动 Gradio 界面
with gr.Blocks(title="迪士尼RAG助手") as demo:
    gr.Markdown("## 🏰 迪士尼乐园问答助手")
    
    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(label="输入你的问题", placeholder="例如：门票多少钱？")
            submit_btn = gr.Button("🔍 提问", variant="primary")
        
        with gr.Column():
            output_answer = gr.Textbox(label="AI 回答", lines=6)
            output_sources = gr.Textbox(label="参考来源", lines=3)
            
    submit_btn.click(
        fn=rag_answer, 
        inputs=input_box, 
        outputs=[output_answer, output_sources]
    )

if __name__ == "__main__":
    print("🚀 服务启动中... 请在浏览器打开下方链接")
    demo.launch()