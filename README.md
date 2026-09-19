# DeepSeek Flash Vision Demo

这是一个基于 Python 的示例项目，演示了如何使用 OpenAI SDK 调用 DeepSeek 的多模态模型（如 `deepseek-flash`）进行图片内容识别和分析。

## 📋 项目简介

本项目展示了如何通过 DeepSeek API 实现图像理解功能。代码通过读取本地图片文件，将其转换为 Base64 编码，并发送给 DeepSeek 模型进行解析，最后输出模型对图片内容的描述，简单来说，它的核心逻辑是：读取本地图片 -> 转码 -> 发送给 AI -> 获取文字描述

## 🛠️ 环境准备

在运行代码之前，请确保你已经安装了必要的依赖库，并配置了环境变量。

### 1. 安装依赖
在命令行（终端）中执行：
pip install openai python-dotenv

### 2. 导入库与环境配置
os, pathlib: 用于处理文件路径和系统环境变量。
base64: 用于将图片文件编码成字符串，因为 API 通常不能直接读取本地文件，需要传输编码后的数据。
dotenv: 用于从 .env 文件中加载密钥，这是一种安全做法，避免将 API Key 硬编码在代码中。
OpenAI: 这里复用了 OpenAI 的官方 SDK。因为 DeepSeek 的 API 接口兼容 OpenAI 格式，所以可以直接用这个库来调用 DeepSeek：
```import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI```


### 3. 初始化客户端
加载配置: 代码尝试从脚本同级目录下的 .env 文件中读取环境变量。
建立连接:
```api_key: 从环境变量中获取 DeepSeek 的密钥。
base_url: 指定 API 的入口地址。如果环境变量没设置，默认使用 https://api.deepseek.com：
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
)```

### 4. 图片处理（核心步骤）
读取: 以二进制只读模式 (rb) 打开名为 test.png 的图片。
编码: 使用 base64 算法将二进制图片数据转换为文本字符串。
解码: .decode("utf-8") 将字节对象转换为标准的 Python 字符串，以便后续拼接到 JSON 数据中：
```with open("test.png", "rb") as f:
    base64_image = base64.b64encode(f.read()).decode("utf-8")```

### 5. 发送请求
模型:  deepseek-flash
消息结构:
这里使用了多模态输入格式。content 是一个列表，包含两部分：
文本部分: 用户的问题“这张图片里有什么？”。
图片部分: 使用 data:image/png;base64,... 这种 Data URL 格式，直接把刚才编码好的图片字符串传给 AI：
```response = client.chat.completions.create(
    model="deepseek-flash",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这张图片里有什么？"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
        }
    ]
)```
### 6. 输出结果
从 API 返回的复杂 JSON 对象中，提取出第一个选择（choices[0]）中的消息内容（即 AI 对图片的描述），并打印到控制台：
```print(response.choices[0].message.content)```

###如何运行
1. 克隆本项目到本地
2. 复制 `.env.example` 为 `.env`，并填入你的 API Key
3. 安装依赖：`pip install openai python-dotenv`
4. 运行脚本：`python "DeepSeek Flash Vision Demo.py"`
5.准备文件: 确保代码同级目录下有一张和test.png的图片。

