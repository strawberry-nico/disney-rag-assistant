# src/build_vector_db.py
import os
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. 配置路径
PERSIST_DIRECTORY = "chroma_db"
SOURCE_DIRECTORY = "processed_texts"
MODEL_PATH = "./models/bge-m3"

def main():
    print("🧠 正在加载 Embedding 模型...")
    try:
        embedding = HuggingFaceEmbeddings(
            model_name=MODEL_PATH,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    except Exception as e:
        print(f"❌ 模型加载失败，请检查路径: {e}")
        return

    # 2. 初始化/连接向量库
    # 注意：这里我们不直接 from_texts，而是先连接库
    if os.path.exists(PERSIST_DIRECTORY):
        print("💾 检测到已有数据库，正在连接...")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding
        )
        # 获取库里已有的来源文件列表
        try:
            existing_data = vectorstore.get()
            # 从 metadata 中提取 source 字段，去重
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

    # 3. 扫描本地文件并过滤
    all_files = glob.glob(os.path.join(SOURCE_DIRECTORY, "*.txt"))
    new_files = []
    
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        # 核心逻辑：如果文件名不在库里，才处理
        if file_name not in existing_sources:
            new_files.append(file_path)
    
    if not new_files:
        print("✅ 没有新文件需要处理，数据库已是最新状态！")
        return

    print(f"📦 发现 {len(new_files)} 个新文件，准备入库...")

    # 4. 加载并切分新文件
    texts = []
    metadatas = []
    
    for file in new_files:
        try:
            with open(file, encoding="utf-8") as f:
                content = f.read()
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
        # 为每个切片复制对应的 metadata
        new_metadatas.extend([metadatas[i]] * len(splits))

    # 5. 增量添加到数据库
    if new_chunks:
        print(f"✂️  生成了 {len(new_chunks)} 个新切片，正在写入向量库...")
        # 关键方法：add_texts (追加) 而不是 from_texts (覆盖)
        vectorstore.add_texts(texts=new_chunks, metadatas=new_metadatas)
        # Chroma 现在的版本通常会自动 persist，但为了保险可以显式调用（虽然新版可能弃用了）
        # vectorstore.persist() 
        print(f"🎉 成功添加 {len(new_files)} 个文件到数据库！")
    else:
        print("⚠️ 文件内容为空或切分失败。")

if __name__ == "__main__":
    main()