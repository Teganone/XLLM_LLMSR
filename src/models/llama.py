
import os
import json
import time
from typing import Dict, Any, Optional, Union, List
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import re
from .base_model import BaseModel
from ..utils.json_utils import JsonUtils
from ..utils.logging_utils import LoggingUtils


logger = LoggingUtils.setup_logger(name="llama_model")
VERIFT_SYSTEM_PROMPT = 'Whether the "statement" can be deduced from the "evidence" logically, answer with only with True or False, do not output other contents.'

class LlamaModel(BaseModel):
    def __init__(self, model_path, params: Dict[str, Any] = None):
        super().__init__(params)
        self.model_path = model_path
        self.default_model_params = {
            "temperature": 0.5,
            "max_new_tokens": 2048,
            "top_p": 0.9,
            "do_sample": True,
            "num_return_sequences": 1
        }
        self.model_params = {**self.default_model_params, **self.params.get("model_params", {})}
        
        self._load_model()


    def _load_model(self):
        """加载模型和分词器"""
        try:
            logger.info(f"正在加载模型: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # 根据可用资源决定是否使用半精度
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"使用设备: {device}")
            
            # 使用pipeline简化模型加载和推理过程
            self.pipe = pipeline(
                "text-generation",
                model=self.model_path,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto",
            )
            
            # 确保tokenizer有正确的pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("模型加载完成！")
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise e




    def invoke(self, messages, **kwargs):
        """
        使用消息格式生成响应
        
        参数:
        - messages: 消息列表，格式为 [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        - **kwargs: 其他参数
        
        返回:
        - 生成的响应
        """
        max_retries = kwargs.pop("max_retries", 3)  # 默认重试3次
        retry_delay = kwargs.pop("retry_delay", 2)  # 默认重试间隔2秒
        for attempt in range(max_retries):
            try:
                # 将消息格式化为Llama模型期望的格式
                prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
                
                # 设置生成参数
                gen_params = self.get_params(**kwargs)
                
                # 生成响应
                outputs = self.pipe(
                    prompt_text,
                    **gen_params
                )
                
                # 提取生成的文本
                generated_text = outputs[0]["generated_text"]
                
                # 提取模型回复（去除输入提示）
                response = generated_text[len(prompt_text):].strip()
                
                return response
            except Exception as e:
                logger.info(f"API调用失败 (尝试 {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logger.info(f"生成响应失败: {e}")
                    return {}

        
  
    
    

if __name__ == '__main__':
    # 初始化模型
    llama_model = LlamaModel(model_path="/datacenter/models/LLM-Research/Llama-3-8B-Instruct")

    # 使用invoke方法
    messages = [
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "请解释什么是机器学习。"}
    ]
    response = llama_model.invoke(messages, temperature=0.7)
    print(response)