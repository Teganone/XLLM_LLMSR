
import os
import json
import time
from typing import Dict, Any, Optional, Union, List
from openai import AzureOpenAI
from dotenv import load_dotenv

from .base_model import BaseModel
from ..utils.json_utils import JsonUtils
from ..utils.logging_utils import LoggingUtils

# 设置日志
logger = LoggingUtils.setup_logger(name="gpt_model")
reasoning_models = ["o1", "o3-mini", "o4-mini", "o3"]

class OpenaiModel(BaseModel):
    """GPT模型封装"""
    
    def __init__(self, model, params: Dict[str, Any] = None):
        super().__init__(params)
        self.model = model
        self.default_model_params = ({"reasoning_effort": "high"}
            if self.model in reasoning_models
            else {"top_p": 0.9, "temperature": 1}
        )
        self.model_params = {**self.default_model_params, **self.config.get("model_params", {})}
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
        try:
            # 获取API参数
            params = self.get_params(**kwargs)
            
            # 调用API
            completion = self.client.chat.completions.create(
                model=self.model,
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
            logger.error(f"生成响应失败: {e}")
            raise e

        
        
    # def generate_parsing_response(self, prompt: str, **kwargs):
    #     for attempt in range(self.max_retries):
    #         try:
    #             message_content = self._invoke_chat_completions(prompt, False, kwargs)
    #             try:
    #                 json_content = json.loads(message_content)
    #                 return json_content
    #             except json.JSONDecodeError:
    #                 print(f"JSON解析失败，尝试修复格式: {message_content[:100]}...")
    #                 return message_content
                    
    #         except Exception as e:
    #             print(f"API调用失败 (尝试 {attempt+1}/{self.max_retries}): {str(e)}")
    #             if attempt < self.max_retries - 1:
    #                 time.sleep(self.retry_delay)
    #             else:
    #                 print("达到最大重试次数，返回空结果")
    #                 return {}
            
    
    # def generate_verification_responsee(self, prompt: str, **kwargs) -> str:
    #     """
    #     获取验证响应
        
    #     Args:
    #         prompt: 用户提示
    #         **kwargs: 其他参数
            
    #     Returns:
    #         验证响应
    #     """
    #     for attempt in range(self.max_retries):
    #         try:
    #             message_content = self._invoke_chat_completions(prompt, True, **kwargs)
    #             return message_content
                    
    #         except Exception as e:
    #             logger.error(f"API调用失败 (尝试 {attempt+1}/{self.max_retries}): {str(e)}")
    #             if attempt < self.max_retries - 1:
    #                 time.sleep(self.retry_delay)
    #             else:
    #                 logger.error("达到最大重试次数，返回空结果")
    #                 return {}
                

    
    # def get_logic_program_response(self, prompt: str, **kwargs) -> str:
    #     """
    #     获取逻辑程序响应
        
    #     Args:
    #         prompt: 提示文本
    #         **kwargs: 其他参数
            
    #     Returns:
    #         生成的逻辑程序
    #     """
    #     # 获取合并后的参数
    #     params = self.get_params(**kwargs)
        
    #     for attempt in range(self.max_retries):
    #         try:
    #             # 构建API参数
    #             api_params = {}
                
    #             # 处理reasoning_effort参数（仅适用于特定模型）
    #             if self.model_name in {"o1", "o3-mini"} and "reasoning_effort" in params:
    #                 api_params["reasoning_effort"] = params["reasoning_effort"]
    #             else:
    #                 api_params["top_p"] = params["top_p"]
    #                 api_params["temperature"] = params["temperature"]
                
    #             completion = self.client.chat.completions.create(
    #                 model=self.model_name,
    #                 messages=[
    #                     {"role": "user", "content": prompt},
    #                 ],
    #                 **api_params
    #             )
                
    #             message = completion.choices[0].message
    #             if message.content:
    #                 # 尝试提取逻辑程序部分
    #                 from ..utils.text_utils import TextUtils
    #                 return TextUtils.extract_logic_program(message.content)
    #             else:
    #                 logger.warning("模型返回空内容")
    #                 return ""
                    
    #         except Exception as e:
    #             logger.error(f"API调用失败 (尝试 {attempt+1}/{self.max_retries}): {str(e)}")
    #             if attempt < self.max_retries - 1:
    #                 time.sleep(self.retry_delay)
    #             else:
    #                 logger.error("达到最大重试次数，返回空结果")
    #                 return ""
                

    def get_params(self, **kwargs):
        params = super().get_params(**kwargs)
        api_params = {} 
        if self.model_name in reasoning_models and "reasoning_effort" in params:
            api_params["reasoning_effort"] = params.get("reasoning_effort", self.model_params["reasoning_effort"])
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
    