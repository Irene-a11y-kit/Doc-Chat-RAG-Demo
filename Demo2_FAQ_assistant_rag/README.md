# 🤖 问答助手 (RAG Demo)

> 一个基于 LangChain 和 DeepSeek 的本地知识库问答系统，演示了从“文档解析”到“智能问答”的完整 RAG（检索增强生成）工作流。

## 📝 项目简介（业务视角）

本项目演示如何读取本地文本文件（Indonesia.txt），利用向量数据库进行语义检索，并结合 DeepSeek 模型回答关于“印度尼西亚旅游景点”的相关问题。

### 🔄 核心工作流
1. **文档加载**：读取本地 `Indonesia.txt` 文件。
2. **文本切分**：按 500 字符切分，保留 50 字符重叠，保证语义连贯。
3. **向量化**：使用 `BAAI/bge-small-zh` 模型将文本转化为向量，存入 Chroma 数据库。
4. **检索问答**：用户提问后，系统先检索最相关的 3 个文本片段，再交给 DeepSeek 大模型生成回答。


## 🛠️ 环境准备

### 1. 安装依赖
在命令行（终端）中执行：
pip install langchain langchain-community langchain-huggingface langchain-openai chromadb python-dotenv sentence-transformers

### 2. 配置环境变量
复制 .env.example 为 .env。
在 .env 文件中填入你的 DeepSeek API Key：
```DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com```

### 3. 准备数据
确保代码同级目录下有一个名为 Indonesia.txt 的文本文件（内容关于印尼旅游介绍）。

🚀 如何运行
直接运行脚本即可：
python rag_demo.py

📂 核心代码解析（防踩坑指南）
1. 镜像加速与静默下载
为了防止从 HuggingFace 下载模型时速度过慢或失败，代码顶部预设了 HF_ENDPOINT 环境变量指向国内镜像源，并关闭了进度条刷屏。
```import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"```

2. 初始化大模型
使用 ChatOpenAI 连接兼容 OpenAI 格式的 DeepSeek 接口。
```from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    temperature=0  # 设为 0 让回答更严谨
)```

3. 构建向量库与检索链
使用 Chroma 作为本地向量数据库，并设置检索器返回最相关的 3 个片段（k=3）。
```from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
vectorstore = Chroma.from_documents(chunks, embeddings)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)```

4. 提问与输出
```result = qa_chain.invoke({"query": "印度尼西亚有什么著名的旅游景点？"})
print(f"🤖 AI 回答：\n{result['result']}")```

💡 常见问题（踩坑记录）
报错 ConnectionError 下载模型失败？
请检查代码顶部的 os.environ["HF_ENDPOINT"] 是否已正确设置为 https://hf-mirror.com。

找不到 Indonesia.txt？
请确保文件名拼写正确，且与 Python 脚本在同一目录下，或者修改 TextLoader 中的路径。

LangChain 版本报错 ModuleNotFoundError: No module named 'langchain.chains'？
这是 LangChain 1.x 版本的变动。请将导入语句改为 from langchain_classic.chains import RetrievalQA。

