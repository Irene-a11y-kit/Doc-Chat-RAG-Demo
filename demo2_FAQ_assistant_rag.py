import os
from pathlib import Path
from dotenv import load_dotenv

# --- 1. 环境配置：静默下载 + 镜像加速 ---
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 这两行让 HuggingFace 不要刷屏打印下载进度条
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 使用新版 HuggingFace 包，消除弃用警告
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA

# --- 2. 加载环境变量 ---
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# --- 3. 初始化大模型 ---
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    temperature=0
)

# --- 4. 加载与切分文档 ---
loader = TextLoader("Indonesia.txt", encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# --- 5. 构建向量库 ---
print("⏳ 正在加载向量模型并构建知识库...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
vectorstore = Chroma.from_documents(chunks, embeddings)

# --- 6. 组装检索问答链 ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# --- 7. 提问与输出 ---
if __name__ == "__main__":
    question = "印度尼西亚有什么著名的旅游景点？"
    
    # 加个漂亮的提示符
    print("\n" + "="*40)
    print(f"🙋‍♂️ 提问：{question}")
    print("="*40)
    
    result = qa_chain.invoke({"query": question})
    
    # 格式化输出
    print(f"\n🤖 AI 回答：\n{result['result']}\n")
    print("-" * 40)
    
    # 可选：如果你想知道 AI 是参考了哪几段原文，可以把下面这几行取消注释
    print("📚 参考的原文片段：")
    for i, doc in enumerate(result['source_documents']):
        print(f"[{i+1}] {doc.page_content[:150]}...\n")