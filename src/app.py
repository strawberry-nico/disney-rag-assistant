import os
import json
import time
import torch
import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatTongyi
from sentence_transformers import CrossEncoder

# 尝试导入 modelscope
try:
    from modelscope.hub.snapshot_download import snapshot_download
except ImportError:
    snapshot_download = None

# --- 1. 基础配置 ---
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
ENABLE_RERANK = True if (USE_GPU and snapshot_download) else False

RERANK_MODEL_ID = "BAAI/bge-reranker-v2-m3"
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
PERSIST_DIRECTORY = "chroma_db"
FEEDBACK_FILE = "user_feedback.jsonl"
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    print("⚠️ 严重警告: 未检测到 DASHSCOPE_API_KEY！")

# --- 2. Prompt 与 逻辑 (保持严谨) ---
REWRITE_PROMPT_TEMPLATE = """
你是一个专业的搜索引擎优化助手。
请将用户的搜索问题重写为 3 个不同角度的搜索关键词，以便在向量数据库中更好地检索。
只需输出关键词，用逗号分隔，不要有任何其他废话。

用户问题：{question}
重写结果：
"""

MICKEY_PROMPT_TEMPLATE = """
你现在是迪士尼乐园的金牌向导“米奇”🐭。
请根据下面的【参考资料】回答游客的问题。
如果资料里没有答案，请委婉告知。
回答要热情、幽默，语气要像米奇一样活泼，最后加上一句神奇的祝福！✨

📖 **参考资料**：
{context}

🗣️ **游客的问题**：
{question}

米奇的回答：
"""

# --- 3. 反馈数据存储模块 ---
def save_feedback(vote_type, question, answer, sources):
    """保存用户反馈到 JSONL 文件"""
    if not question or not answer:
        return "⚠️ 还没有对话内容哦"
    
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vote": vote_type,
        "question": question,
        "answer": answer,
        "sources": sources,
        "model_config": "Rerank-v1.2" if ENABLE_RERANK else "CPU-Lite"
    }
    
    try:
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        return f"✅ 已记录您的反馈 ({'👍' if vote_type=='up' else '👎'})，米奇收到啦！"
    except Exception as e:
        return f"❌ 保存失败: {e}"

# --- 4. 模型加载 (保持不变) ---
print(f"🖥️  环境: {DEVICE} | Rerank: {ENABLE_RERANK}")
try:
    path = snapshot_download(EMBEDDING_MODEL_ID) if snapshot_download else EMBEDDING_MODEL_ID
    embedding = HuggingFaceEmbeddings(model_name=path, model_kwargs={"device": DEVICE}, encode_kwargs={"normalize_embeddings": True})
except: embedding = None

reranker = None
if ENABLE_RERANK:
    try:
        path = snapshot_download(RERANK_MODEL_ID)
        reranker = CrossEncoder(path, device=DEVICE)
    except: ENABLE_RERANK = False

vectorstore = None
if os.path.exists(PERSIST_DIRECTORY) and embedding:
    vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)

# --- 5. 核心逻辑 ---
def rag_pipeline(query):
    if not query.strip(): return "", "", "" # 增加一个空返回给隐藏的state
    
    # Rewrite & Recall
    queries = [query]
    if DASHSCOPE_API_KEY:
        try:
            llm = ChatTongyi(model="qwen-max", api_key=DASHSCOPE_API_KEY)
            res = llm.invoke(REWRITE_PROMPT_TEMPLATE.format(question=query))
            queries.extend([q.strip() for q in res.content.split(',')])
        except: pass
    
    top_k = 50 if ENABLE_RERANK else 3
    candidates = []
    if vectorstore:
        for q in list(set(queries)):
            candidates.extend(vectorstore.similarity_search(q, k=top_k))
    
    # Deduplicate
    unique_docs = {d.page_content: d for d in candidates}
    docs = list(unique_docs.values())

    # Rerank
    if ENABLE_RERANK and reranker and docs:
        pairs = [[query, d.page_content] for d in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        docs = [d for d, s in ranked[:3]]
    else:
        docs = docs[:3]
    
    # Generate
    if not docs: return "❌ 抱歉，米奇没找到相关信息。", "", query
    
    context = "\n\n".join([d.page_content for d in docs])
    try:
        llm = ChatTongyi(model="qwen-max", api_key=DASHSCOPE_API_KEY, temperature=0.7)
        resp = llm.invoke(MICKEY_PROMPT_TEMPLATE.format(context=context, question=query))
        answer = resp.content
    except Exception as e: answer = f"❌ Error: {e}"
    
    sources = "\n".join([f"📄 {os.path.basename(d.metadata.get('source','未知'))}" for d in docs])
    
    # 返回: 回答, 来源, 原问题(用于反馈)
    return answer, sources, query

# --- 5. ✨ UI 美化重构区 ✨ ---

# 定制迪士尼主题 (红色主调，圆润风格)
theme = gr.themes.Soft(
    primary_hue="red",
    secondary_hue="yellow",
    neutral_hue="slate",
    radius_size="lg"
).set(
    button_primary_background_fill="#FF4B4B",
    button_primary_background_fill_hover="#FF2424",
    button_primary_text_color="white",
    block_title_text_color="#FF4B4B"
)

# 自定义 CSS 增加氛围感
css = """
.gradio-container {background-color: #FAFAFA}
h1 {text-align: center; color: #FF4B4B; font-family: 'Comic Sans MS', sans-serif;}
.feedback-btn {font-size: 14px !important;}
"""

with gr.Blocks(theme=theme, css=css, title="Disney RAG Pro") as demo:
    
    # 顶部标题栏
    gr.HTML("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="font-size: 2.5em; margin-bottom: 10px;">🏰 迪士尼魔法助手</h1>
        <p style="font-size: 1.2em; color: #666;">
            我是米奇 🐭，你的专属私人导游！(Powered by <b>Rerank</b> & <b>Qwen</b>)
        </p>
    </div>
    """)

    with gr.Row():
        # === 左侧：操作区 ===
        with gr.Column(scale=4):
            inp = gr.Textbox(
                label="✨ 请输入你的问题", 
                placeholder="例如：那个骑摩托车的项目叫什么？",
                lines=3,
                show_label=True
            )
            
            with gr.Row():
                btn_clear = gr.Button("🗑️ 清空", variant="secondary")
                btn_submit = gr.Button("🚀 魔法提问", variant="primary", scale=2)
            
            # 快捷示例 (录屏神器！)
            gr.Examples(
                examples=[
                    ["创极速光轮刺激吗？"],
                    ["带5岁的小孩去哪里玩比较好？"],
                    ["迪士尼乐园几点开门？"],
                    ["加勒比海盗排队久吗？"]
                ],
                inputs=inp,
                label="💡 试一试这些问题"
            )

        # === 右侧：展示区 ===
        with gr.Column(scale=5):
            # 这里的 state 用于暂存“当前正在问的问题”，方便传给反馈按钮
            current_question = gr.State()
            
            out_ans = gr.Markdown(label="米奇的回答")
            
            # 来源折叠起来，保持界面整洁
            with gr.Accordion("📚 查看知识来源 (Rerank Top-3)", open=False):
                out_src = gr.Textbox(label="来源文档", lines=3, show_label=False)
            
            # 反馈区
            with gr.Row():
                gr.Markdown("📝 **觉得这个回答怎么样？**")
                btn_like = gr.Button("👍 很有用", size="sm")
                btn_dislike = gr.Button("👎 不太准", size="sm")
            
            feedback_msg = gr.Markdown(visible=True)

    # --- 事件绑定 ---
    # 提交问题
    btn_submit.click(
        fn=rag_pipeline,
        inputs=inp,
        outputs=[out_ans, out_src, current_question] # 同时更新问题到 State
    )
    # 回车提交
    inp.submit(
        fn=rag_pipeline,
        inputs=inp,
        outputs=[out_ans, out_src, current_question]
    )
    # 清空
    btn_clear.click(lambda: ("", "", ""), outputs=[inp, out_ans, out_src])

    # 反馈逻辑
    btn_like.click(
        fn=lambda q, a: save_feedback("up", q, a, ""),
        inputs=[current_question, out_ans],
        outputs=feedback_msg
    )
    btn_dislike.click(
        fn=lambda q, a: save_feedback("down", q, a, ""),
        inputs=[current_question, out_ans],
        outputs=feedback_msg
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006, share=False)