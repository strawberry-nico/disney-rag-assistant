import os
import glob
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 尝试导入 modelscope，用于云端极速下载
try:
    from modelscope.hub.snapshot_download import snapshot_download
except ImportError:
    snapshot_download = None

# --- 1. 配置路径与参数 ---
PERSIST_DIRECTORY = "chroma_db"
SOURCE_DIRECTORY = "processed_texts"
LOCAL_MODEL_PATH = "./models/bge-m3"  # 本地路径
ONLINE_MODEL_ID = "BAAI/bge-m3"       # 线上 ID

def main():
    # --- 2. 硬件与模型自适应加载 ---
    # 自动检测 GPU
    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    print(f"\n" + "="*40)
    print(f"🖥️  构建设备: {device.upper()}")
    
    # 智能选择模型路径
    model_name_or_path = ONLINE_MODEL_ID # 默认用在线 ID
    
    if os.path.exists(LOCAL_MODEL_PATH):
        print(f"📂 发现本地模型: {LOCAL_MODEL_PATH}")
        model_name_or_path = LOCAL_MODEL_PATH
    else:
        print(f"🌐 本地模型不存在，准备从云端加载: {ONLINE_MODEL_ID}")
        # 如果在 AutoDL (装了 modelscope)，则使用极速下载
        if snapshot_download:
            try:
                print("🚀 [AutoDL] 正在通过 ModelScope 极速下载...")
                model_name_or_path = snapshot_download(ONLINE_MODEL_ID)
                print(f"✅ 下载完成，路径: {model_name_or_path}")
            except Exception as e:
                print(f"⚠️ ModelScope 下载异常，尝试直接加载: {e}")

    print(f"🧠 正在加载 Embedding 模型 (Device={device})...")
    try:
        embedding = HuggingFaceEmbeddings(
            model_name=model_name_or_path,
            model_kwargs={"device": device}, # 👈 关键：这里换成了 GPU
            encode_kwargs={"normalize_embeddings": True}
        )
    except Exception as e:
        print(f"❌ 模型加载彻底失败: {e}")
        return

    # --- 3. 初始化/连接向量库 (保留你优秀的增量逻辑) ---
    if os.path.exists(PERSIST_DIRECTORY):
        print("💾 检测到已有数据库，正在连接...")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding
        )
        # 获取库里已有的来源文件列表
        try:
            existing_data = vectorstore.get()
            existing_sources = set()
            if existing_data and 'metadatas' in existing_data:
                for meta in existing_data['metadatas']:
                    if meta and 'source' in meta:
                        existing_sources.add(meta['source'])
            print(f"👀 库里已有 {len(existing_sources)} 个文档。")
        except Exception:
            existing_sources = set()
    else:
        print("🆕 未找到数据库，将创建新库...")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding
        )
        existing_sources = set()

    # --- 4. 扫描并过滤新文件 ---
    if not os.path.exists(SOURCE_DIRECTORY):
        print(f"❌ 错误: 找不到 {SOURCE_DIRECTORY} 文件夹！请先上传数据。")
        return

    all_files = glob.glob(os.path.join(SOURCE_DIRECTORY, "*.txt"))
    new_files = []
    
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        if file_name not in existing_sources:
            new_files.append(file_path)
    
    if not new_files:
        print("✅ 没有新文件需要处理，数据库已是最新状态！")
        return

    print(f"📦 发现 {len(new_files)} 个新文件，准备处理...")

    # --- 5. 加载与切分 ---
    texts = []
    metadatas = []
    
    for file in new_files:
        try:
            with open(file, encoding="utf-8") as f:
                content = f.read()
            if not content.strip(): continue # 跳过空文件
            texts.append(content)
            metadatas.append({"source": os.path.basename(file)})
        except Exception as e:
            print(f"⚠️ 跳过文件 {file}: {e}")

    # 切分器配置
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )

    new_chunks = []
    new_metadatas = []

    for i, text in enumerate(texts):
        splits = splitter.split_text(text)
        new_chunks.extend(splits)
        new_metadatas.extend([metadatas[i]] * len(splits))

    # --- 6. 写入数据库 ---
    if new_chunks:
        print(f"✂️  生成了 {len(new_chunks)} 个切片，正在写入向量库...")
        vectorstore.add_texts(texts=new_chunks, metadatas=new_metadatas)
        print(f"🎉 成功添加 {len(new_files)} 个文件到数据库！")
        print("="*40 + "\n")
    else:
        print("⚠️ 有文件但没切分出内容，请检查文件格式。")

if __name__ == "__main__":
    main()