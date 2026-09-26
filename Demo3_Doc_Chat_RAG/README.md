# Doc-Chat-RAG-Demo

一个能吃 PDF、能回答、能溯源来源的本地 RAG 工具。
基于 **LangChain + DeepSeek + FAISS + BGE** 实现，跑通「PDF 解析 → 文本切分 → 向量化 → 检索增强问答」完整链路。

- **Embedding**：HuggingFace `bge-small-zh-v1.5`（中文向量模型，本地运行）
- **LLM**：DeepSeek Chat（`deepseek-chat`）
- **向量库**：FAISS（本地，无需外部服务）
- **编排**：LangChain（LCEL）

## 效果示例

![运行截图](demo.png)

**提问**：出售假冒商品扣多少分？

**回答**：
> 出售假冒商品的扣分要看具体情况：
> - 情节特别严重的，每次扣 48 分；
> - 情节严重的，每次扣 24 分；
> - 通过信息层面判断的，每件扣 2 分（3 天内不超过 12 分）；
> - 实际出售的，每次扣 12 分；
> - 为出售假冒、盗版商品提供便利条件的，每次扣 2 分，情节严重的每次扣 12 分。
> （来源：sample.pdf 第8页）

**提问**：淘宝卖家可以发顺丰快递吗？（文档中没有）

**回答**：
> 抱歉，知识库中没有相关信息。

> 以上四轮问答完整截图：`demo.png`
> —— 前三轮证明"能查到"，最后一轮证明"查不到不编造"。

![运行截图](demo.png)

## 项目简介

本项目演示如何用 RAG 技术让大模型「读懂」一份 PDF 并基于文档内容回答问题。

完整流程：读取本地 PDF → 递归切分文本 → 用中文向量模型 `bge-small-zh-v1.5` 向量化并写入 FAISS → 检索 Top-K 相关片段拼入 Prompt → 交给 DeepSeek 生成基于原文的回答，并要求「文档里没有的就直说，不编造」。

所有路径基于脚本位置动态定位，开箱即用、可移植。

## 项目架构

```text
用户提问 (Question)
       |
       v
+----------------------------------+
|        RAG 问答链 (LangChain)    |
|                                  |
|  1. 检索器 (Retriever)           |
|     -> 从 FAISS 向量库中召回     |
|        Top-K 相关文本片段        |
|                                  |
|  2. 格式化 (Format)              |
|     -> 拼接来源、页码与原文      |
|                                  |
|  3. 提示词工程 (Prompt)          |
|     -> 注入【参考资料】与        |
|        【用户问题】              |
|                                  |
|  4. 大模型生成 (LLM)             |
|     -> DeepSeek Chat 基于资料    |
|        生成口语化、准确的回答    |
+----------------------------------+
       |
       v
   最终回答 (Answer)

**现象**：新建 `rag_env` 后装包，比旧环境快很多。
**原因**：pip 会把下载过的 `.whl` 缓存在本地（`pip cache dir` 可查看），新环境下装同一个包时直接复用缓存，无需重新下载。
**启发**：缓存是工程效率的关键，企业级系统里 CDN、Docker 层缓存、模型缓存都是同一逻辑。

## 项目结构
.
|-- main.py            # 主程序入口
|-- .env               # 环境变量（API Key，不上传）
|-- requirements.txt   # 依赖清单
|-- README.md          # 项目说明
|-- sample.pdf         # 待问答的示例 PDF
|-- models/            # 自动下载的向量模型（不上传）
|-- vector_db/         # 生成的 FAISS 向量库（不上传）
```

## 常见问题

- **首次运行很慢 / 下载失败？** 模型会走 `hf-mirror.com` 镜像自动下载，需保持联网；下载完成后会缓存到本地，无需重复下载。
- **CPU 还是 GPU？** 默认使用 `faiss-cpu` 且 `device="cpu"`，无需显卡即可运行；如需加速可改用 `faiss-gpu`。
- **回答不准确？** 可调整 `chunk_size`、`chunk_overlap` 以及检索的 `k` 值来优化召回质量。

## 技术栈
模块	选型	说明
文档解析	PyPDFLoader	将 PDF 解析为带页码元数据的文档对象
文本切分	RecursiveCharacterTextSplitter	中文友好、支持重叠切分
向量化	bge-small-zh-v1.5	中文语义向量模型，本地运行
向量检索	FAISS	高性能本地向量库，无需外部服务
大模型	deepseek-chat	通过 OpenAI 兼容协议调用
编排框架	LangChain (LCEL)	链式编排检索、Prompt、模型调用

## 快速开始
0. 克隆项目
```
git clone https://github.com/Irene-a11y-kit/Doc-Chat-RAG-Demo.git
cd Doc-Chat-RAG-Demo```

1. 创建虚拟环境
```
conda create -n rag_env python=3.11 -y
conda activate rag_env
```

2. 安装依赖
```pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple```
*torch 需 ≥ 2.6.0，否则会因 torch.load 安全漏洞被 transformers 拦截。

3. 配置环境变量
复制 .env.example 为 .env，填入你的 DeepSeek API Key：
```
DEEPSEEK_API_KEY=sk-你的API_KEY_在这里
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

4. 准备素材
把要问答的 PDF 放到项目根目录，命名为 sample.pdf。
首次运行会自动下载向量模型到 models/，并生成向量库到 vector_db/，之后运行会直接复用。

5. 运行
```python_main.py```

## 代码解析
1. 导入库与环境配置
```
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
```
pathlib：BASE_DIR = Path(__file__).resolve().parent 基于脚本位置定位，保证项目在任何电脑、任意工作目录下都能正确找到模型和向量库，这是「可移植」的关键。
dotenv：从 .env 加载 API Key，避免密钥硬编码，是安全最佳实践。
PyPDFLoader：把 PDF 解析为带页码元数据的文档对象。
RecursiveCharacterTextSplitter：按分隔符优先级（段落 → 句子 → 标点）递归切分长文本，保留 chunk_overlap 重叠，避免语义在切分处被截断。
HuggingFaceEmbeddings：加载中文向量模型，把文本转成向量。
FAISS：Facebook 开源的高性能向量检索库，本地存储，无需外部服务。
ChatOpenAI：复用 OpenAI SDK 协议调用 DeepSeek（接口高度兼容），无需专用库。
LCEL 组件：RunnablePassthrough / ChatPromptTemplate / StrOutputParser，把「检索 → 填充 Prompt → 调用模型 → 解析输出」串成一条流水线。

2. 模型与向量化（自动下载）
```os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")```

3. PDF 解析与文本切分
load_pdf() 做了防御性校验（文件是否存在、后缀是否为 .pdf）；split_documents() 针对中文优化了分隔符（。！？，），chunk_size=500、chunk_overlap=50 在召回精度与上下文完整性之间取得平衡。

4. 向量库构建 / 加载
build_vector_db() 负责「解析 → 切分 → 向量化 → 落盘」；load_vector_db() 在已存在向量库时直接加载，避免每次运行都重建。代码还会比较 PDF 与向量库的修改时间，PDF 比库新时自动重建。

5.RAG 问答链
format_docs() 在拼接检索片段时附带来源文件名与页码，方便回答末尾标注出处。Prompt 明确了「基于资料回答 / 没有就说没有 / 不编造 / 口语化」四条约束，最后通过 LCEL 管道 | 把整条链组装起来。

## 项目结构
.
|-- main.py            # 主程序入口
|-- check_env.py       # 环境体检脚本
|-- requirements.txt   # 依赖清单
|-- .env.example       # 环境变量模板
|-- .gitignore
|-- README.md
|-- docs/
|   └── demo.png       # 效果截图
|-- sample.pdf         # 待问答的示例 PDF（自备）
|-- models/            # 向量模型（自动下载，不上传）
|-- vector_db/         # FAISS 向量库（自动生成，不上传）

## 踩坑记录
环境搭建与模型加载过程中踩过的坑，记录如下。

1. Windows 11 SAC 拦截 PyTorch DLL
现象：运行时报 OSError: [WinError 4551] 应用程序控制策略已阻止此文件。
原因：Windows 11「智能应用控制（Smart App Control）」拦截了未签名的 PyTorch DLL。
解决：关闭 SAC（Windows 安全中心 → 应用和浏览器控制 → 智能应用控制 → 关闭）。

2. HuggingFace SSL 证书验证失败
现象：下载模型时报 [SSL: CERTIFICATE_VERIFY_FAILED]。
原因：本地网络对 HTTPS 流量做拦截检查，Python 无法验证证书链。
解决：优先使用本地模型文件，避免运行时联网；手动下载可用 hf-mirror.com 镜像。

3. torch < 2.6 安全限制
现象：加载模型时报 ValueError: Due to a serious vulnerability issue in torch.load...。
原因：torch < 2.6 存在 torch.load 安全漏洞，新版 transformers 要求升级。
```pip install "torch>=2.6.0" --index-url https://download.pytorch.org/whl/cpu```

4. LangChain 1.x 路径变动
现象：ModuleNotFoundError: No module named 'langchain.text_splitter'。
原因：LangChain 1.x 把组件拆到独立包，路径变了。
```
# 旧
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
# 新
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
```

5. 模型文件夹结构不对导致重复下载
现象：本地已有模型文件，但代码仍尝试联网下载，触发 SSL 报错。
原因：代码查找 models/bge-small-zh-v1.5/，但文件被直接放在 models/ 下，多套/少套了一层。
解决：确保结构为：
models/
└── bge-small-zh-v1.5/
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer.json

6. 改了 PDF，但答案还是旧的
现象：替换了 sample.pdf，但问答结果还是上一份文档的内容。
原因：向量库只在不存在时构建。换了 PDF 但 vector_db/ 还在，代码会直接加载旧库。
解决：删除 vector_db/ 重建；当前代码已加入"PDF 比库新则自动重建"的逻辑，正常情况下无需手动删。

## 常见问题
首次运行很慢 / 下载失败？ 模型走 hf-mirror.com 镜像自动下载，需保持联网；下载完成后缓存到本地，无需重复下载。
CPU 还是 GPU？ 默认 faiss-cpu 且 device="cpu"，无需显卡；如需加速可改用 faiss-gpu。
回答不准确？ 可调整 chunk_size、chunk_overlap 以及检索的 k 值来优化召回质量。

## 后续计划
□ 支持多文档上传与跨文档检索
□ 调优切片策略（chunk size / overlap / k 值）并做对比实验
□ 加 Streamlit 前端，做成可交互 Demo

License

MIT

[def]: demo-1.png
