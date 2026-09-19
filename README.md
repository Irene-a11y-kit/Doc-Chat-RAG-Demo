# DeepSeek Flash Vision Demo
这是一个基于 Python 的示例项目，演示了如何使用 OpenAI SDK 调用 DeepSeek 的多模态模型进行图片内容识别和分析。
核心逻辑：读取本地图片 -> 转码 -> 发送给 AI -> 获取文字描述。

## 环境准备

在运行代码之前，请确保你已经安装了必要的依赖库。

安装依赖命令：pip install openai python-dotenv

## Python 代码参考

第一步，导入库：
import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

第二步，初始化客户端：
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))

第三步，处理图片并发送请求：
with open("test.png", "rb") as f:
    base64_image = base64.b64encode(f.read()).decode("utf-8")
response = client.chat.completions.create(
    model="deepseek-flash",
    messages=[{"role": "user", "content": [{"type": "text", "text": "这张图片里有什么？"}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}]
)

第四步，输出结果：
print(response.choices[0].message.content)
