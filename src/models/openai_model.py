
import os
import json
import time
from typing import Dict, Any, Optional, Union, List
from openai import AzureOpenAI
from dotenv import load_dotenv

from src.models.base_model import BaseModel
from src.utils.json_utils import JsonUtils
from src.utils.logging_utils import LoggingUtils

# 设置日志
logger = LoggingUtils.setup_logger(
    name="openai_model",
    log_file="logs/openai.log"
    )

reasoning_models = ["o1", "o3-mini", "o4-mini", "o3"]

class OpenaiModel(BaseModel):
    """GPT模型封装"""
    
    def __init__(self, model, params: Dict[str, Any] = {}):
        super().__init__(params)
        self.model_name = model
        self.default_model_params = ({"reasoning_effect": "high", 'temperature':1}
            if self.model_name in reasoning_models
            else {"top_p": 0.9, "temperature": 0.5}
        )
        self.model_params = {**self.default_model_params, **self.params}
        load_dotenv(dotenv_path='.env')
        
        try:
            self.client = AzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ['AZURE_OPENAI_API_KEY'],
                api_version=os.environ['OPENAI_API_VERSION'],
            )
            
            logger.info(f"GPT模型初始化完成，使用模型: {self.model_name}")
            
        except KeyError as e:
            logger.error(f"环境变量缺失: {e}")
            logger.error("请确保已设置以下环境变量:")
            logger.error("- AZURE_OPENAI_ENDPOINT")
            logger.error("- AZURE_OPENAI_API_KEY")
            logger.error("- OPENAI_API_VERSION")
            raise
        except Exception as e:
            logger.error(f"GPT模型初始化失败: {e}")
            raise
    
   
    def invoke(self, messages:List[Dict[str, str]], **kwargs):
        """
        使用消息格式生成响应
        
        参数:
        - messages: 消息列表，格式为 [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        - **kwargs: 其他参数
        
        返回:
        - 生成的响应
        """
        max_retries = kwargs.pop("max_retries", 3)  # 默认不重试
        retry_delay = kwargs.pop("retry_delay", 2)  # 默认重试间隔2秒
        for attempt in range(max_retries):

            try:
                # 获取API参数
                params = self.get_params(**kwargs)
                # 调用API
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **params
                )
                
                message = completion.choices[0].message
                if message.content:
                    return message.content
                else:
                    logger.warning("模型返回空内容")
                    return {}
            except Exception as e:
                logger.info(f"API调用失败 (尝试 {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logger.info(f"生成响应失败: {e}")
                    return {}

        

    def get_params(self, **kwargs):
        params = super().get_params(**kwargs)
        api_params = {} 
        if self.model_name in reasoning_models:
            api_params["reasoning_effect"] = params.get("reasoning_effect", self.model_params["reasoning_effect"])
        else:
            api_params["top_p"] = params.get("top_p", self.model_params["top_p"])
            api_params["temperature"] = params.get("temperature",self.model_params["temperature"])
        
        if "max_tokens" in params:
            api_params["max_tokens"] = params["max_tokens"]
        if "frequency_penalty" in params:
            api_params["frequency_penalty"] = params["frequency_penalty"]
        if "presence_penalty" in params:
            api_params["presence_penalty"] = params["presence_penalty"]
        return api_params
    
if __name__ == '__main__':
    openai_model = OpenaiModel(model="gpt-4")

    # 使用invoke方法
    messages = [
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "请解释什么是机器学习。"}
    ]
    response = openai_model.invoke(messages, temperature=0.7)
    print(response)