import os
import sys
import json
import time
from tqdm import tqdm
from openai import AzureOpenAI
from dotenv import load_dotenv


load_dotenv(dotenv_path='.env')

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ['AZURE_OPENAI_API_KEY'],
    api_version=os.environ['OPENAI_API_VERSION'],
)

MODEL = "o3"


def get_validation_response(prompt, text, model="o3", temperature=1, max_retries=3, retry_delay=2):
    """
    使用Azure OpenAI API获取响应
    
    Args:
        prompt: 系统提示
        text: 用户输入
        model: 模型名称
        temperature: 温度参数
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
    
    Returns:
        模型响应内容
    """
    
    for attempt in range(max_retries):
        try:
            reasoning_params = (
                {"reasoning_effort": "high"}
                if model in {"o1", "o3-mini", "o3", "o4-mini"}
                else {"top_p": 0.9, "temperature": 0.5}
            )
            print(reasoning_params)
            completion = client.chat.completions.create(
                model=model,
                # reasoning_effort="high",
                **reasoning_params,
                temperature=temperature,
                messages=[
                    # {"role": "user", "content": prompt},
                    {"role": "system", "content": f"{prompt}" },
                    {"role": "user", "content": f"{text}"},
                    
                ],
                # response_format={"type": "json_object"}
            )
            
            message = completion.choices[0].message
            if message.content:
                return message.content
            else:
                print("模型返回空内容")
                return {}
                
        except Exception as e:
            print(f"API调用失败 (尝试 {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print("达到最大重试次数，返回空结果")
                return {}


