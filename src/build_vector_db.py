# src/build_vector_db.py
import os
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

def load_documents():
    texts, metadatas = [], []
    for file in glob.glob("processed_texts/*.txt"):
        with open(file, encoding="utf-8") as f:
            texts.append(f.read())
        metadatas.append({"source": os.path.basename(file)})
    return texts, metadatas

def split_texts(texts, metadatas):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=[
            "\n\n", "\n", 
            "。", "！", "？", "；", "……",
            "”", "“",
            " ", ""
        ]
    )
    chunks, chunk_metadatas = [], []
    for i, text in enumerate(texts):
        splits = splitter.split_text(text)
        chunks.extend(splits)
        chunk_metadatas.extend([metadatas[i]] * len(splits))
    return chunks, chunk_metadatas

def main():
    print("🔍 加载文档...")
    texts, metadatas = load_documents()
    if not texts:
        print("❌ processed_texts/ 为空！请先运行 parse_docs.py")
        return
    
    print("✂️ 切分文本...")
    chunks, chunk_metadatas = split_texts(texts, metadatas)
    
    print("🧠 加载 BGE-M3（首次运行会下载 ～2.2GB）...")
    embedding = HuggingFaceBgeEmbeddings(
        model_name="./models/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        query_instruction="为这个句子生成表示以用于检索相关文章："
    )
    
    print("💾 构建 Chroma 向量库...")
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embedding,
        metadatas=chunk_metadatas,
        persist_directory="chroma_db"
    )
    vectorstore.persist()
    print("🎉 向量库已保存到 chroma_db/")

if __name__ == "__main__":
    main()