"""
Doc-Chat-RAG-Demo | PDF 解析 + RAG 问答
Embedding: HuggingFace bge-small-zh-v1.5 (本地自动下载)
LLM: DeepSeek Chat
向量库: FAISS
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ============================================================
# 0. 环境配置
# ============================================================
load_dotenv()
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 基于脚本位置定位（关键：可移植）
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "bge-small-zh-v1.5"
DB_DIR = BASE_DIR / "vector_db"

# ============================================================
# 1. 模型配置
# ============================================================
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    temperature=0.3,
)


def get_embeddings():
    """本地没有模型就自动下载，然后返回 Embedding 实例"""
    if not MODEL_DIR.exists():
        print(f"[模型] 本地不存在，自动下载到: {MODEL_DIR}")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="BAAI/bge-small-zh-v1.5",
            local_dir=str(MODEL_DIR),
        )

    return HuggingFaceEmbeddings(
        model_name=str(MODEL_DIR),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


embeddings = get_embeddings()


# ============================================================
# 2. PDF 解析
# ============================================================
def load_pdf(file_path: str):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"文件不是 PDF: {file_path}")
    return PyPDFLoader(str(path)).load()


def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "，", ".", " ", ""],
        add_start_index=True,
    )
    return splitter.split_documents(documents)


# ============================================================
# 3. 向量库构建 / 加载
# ============================================================
def build_vector_db(pdf_path: str, db_path: str = None):
    db_path = db_path or str(DB_DIR)
    print(f"[知识库构建] 加载 PDF: {pdf_path}")
    documents = load_pdf(pdf_path)
    print(f"[知识库构建] 原始页数: {len(documents)}")

    print(f"[知识库构建] 切分文本...")
    chunks = split_documents(documents)
    print(f"[知识库构建] 切片数量: {len(chunks)}")

    print(f"[知识库构建] 向量化并写入 FAISS...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(db_path)
    print(f"[知识库构建] 已保存到: {db_path}")
    return vectorstore


def load_vector_db(db_path: str = None):
    db_path = db_path or str(DB_DIR)
    if not Path(db_path).exists():
        raise FileNotFoundError(f"向量库不存在: {db_path}，请先构建")
    print(f"[知识库加载] 从 {db_path} 加载")
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)


# ============================================================
# 4. RAG 问答链
# ============================================================
def format_docs(docs):
    # ✅ 改动：source 只取文件名，不暴露本地绝对路径
    return "\n\n".join(
        f"[来源: {Path(d.metadata.get('source', '未知')).name} "
        f"第{d.metadata.get('page', '?')}页]\n{d.page_content}"
        for d in docs
    )


def create_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    rag_prompt = ChatPromptTemplate.from_template("""
你是一个专业的文档问答助手，请严格基于以下参考资料回答用户问题。

【参考资料】
{context}

【用户问题】
{question}

【回答要求】
1. 如果参考资料中有答案，请简洁、准确地回答。
2. 如果参考资料中没有相关信息，请直接说"抱歉，知识库中没有相关信息"，不要编造。
3. 回答要口语化、友好。
4. 如果引用了具体来源，可以在结尾用括号标注（如：来源：文件名 第X页）。
""")

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )


# ============================================================
# 5. 主入口
# ============================================================
if __name__ == "__main__":
    # ✅ 改动：启动前先检查 Key
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("[错误] 未检测到 DEEPSEEK_API_KEY")
        print("[提示] 请在项目根目录创建 .env，填入：")
        print("        DEEPSEEK_API_KEY=sk-你的Key")
        raise SystemExit(1)

    PDF_PATH = BASE_DIR / "sample.pdf"
    DB_PATH = str(DB_DIR)

    if not Path(DB_PATH).exists():
        if not PDF_PATH.exists():
            print(f"[错误] 未找到 PDF 文件: {PDF_PATH}")
            print(f"[提示] 请把待问答的 PDF 命名为 sample.pdf 放到项目根目录")
            raise SystemExit(1)
        vectorstore = build_vector_db(str(PDF_PATH), DB_PATH)
    else:
        pdf_mtime = PDF_PATH.stat().st_mtime if PDF_PATH.exists() else 0
        db_mtime = Path(DB_PATH).stat().st_mtime
        if pdf_mtime > db_mtime:
            print("[提示] 检测到 PDF 已更新，自动重建向量库...")
            vectorstore = build_vector_db(str(PDF_PATH), DB_PATH)
        else:
            vectorstore = load_vector_db(DB_PATH)

    print("[调试] 向量库来源:", set(
    d.metadata.get("source", "未知")
    for d in vectorstore.docstore._dict.values()
    ))    

    chain = create_rag_chain(vectorstore)

    # ✅ 改动：改成命令行循环，可以连续提问
    print("\n=== 知识库已就绪，输入问题开始问答（输入 q 退出）===")
    while True:
        question = input("\n问题: ").strip()
        if question.lower() in ("q", "quit", "exit"):
            print("再见。")
            break
        if not question:
            continue
        answer = chain.invoke(question)
        print(f"\n回答: {answer}")