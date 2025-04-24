
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
            "do_sample": True
        }
        self.model_params = {**self.default_model_params, **self.config.get("model_params", {})}
        
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


    def _invoke_chat_completions(self, prompt: str, verify: bool = False, **kwargs):
        """
        调用模型生成响应
        
        Args:
            prompt: 提示文本
            verify: 是否是验证任务
            **kwargs: 其他参数
            
        Returns:
            生成的文本
        """
        # 构建Llama格式的提示
        messages = self._set_message(prompt, verify)
        
        # 将消息格式化为Llama模型期望的格式
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        
        # 设置生成参数
        gen_params = {
            "max_new_tokens": 1024,
            "temperature": 0.5,
            "do_sample": True,
            "top_p": 0.9,
            "num_return_sequences": 1,
        }
        
        # 验证任务使用特定参数
        if verify:
            gen_params["max_new_tokens"] = 128
            gen_params["temperature"] = 0.1
            gen_params["do_sample"] = False
        
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
    
     
    def _set_message(self, prompt, verify=False):
        """
        设置消息格式
        
        Args:
            prompt: 提示文本
            verify: 是否是验证任务
            
        Returns:
            消息列表
        """
        if verify:
            # 验证任务使用系统提示
            return [
                {"role": "system", "content": VERIFT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        else:
            # 解析任务只使用用户提示
            return [
                {"role": "user", "content": prompt}
            ]
    
    def generate_parsing_response(self, prompt: str, **kwargs):
        """
        生成解析响应
        
        Args:
            prompt: 提示文本
            **kwargs: 其他参数
            
        Returns:
            解析结果（通常是JSON格式）
        """
        for attempt in range(self.max_retries):
            try:
                # 调用模型生成响应
                message_content = self._invoke_chat_completions(prompt, False, **kwargs)
                
                # 尝试解析JSON
                try:
                    json_content = JsonUtils.extract_json_from_text(message_content)
                    if isinstance(json_content, dict):
                        return json_content
                    else:
                        logger.warning(f"无法提取有效JSON: {message_content[:100]}...")
                        return message_content
                except Exception as e:
                    logger.error(f"JSON处理失败: {str(e)}")
                    return message_content
                    
            except Exception as e:
                logger.error(f"模型调用失败 (尝试 {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error("达到最大重试次数，返回空结果")
                    return {}
    
    def generate_verification_response(self, prompt: str, **kwargs):
        """
        生成验证响应
        
        Args:
            prompt: 用户提示
            **kwargs: 其他参数
            
        Returns:
            验证结果
        """
        for attempt in range(self.max_retries):
            try:
                # 调用模型生成响应，使用验证模式
                message_content = self._invoke_chat_completions(prompt, True, **kwargs)
                
                # 处理验证响应
                if not isinstance(message_content, str):
                    return "false"
                
                true_match = re.search(r'\btrue\b', message_content.lower())
                false_match = re.search(r'\bfalse\b', message_content.lower())
                
                if true_match and not false_match:
                    return "true"
                elif false_match and not true_match:
                    return "false"
                elif true_match and false_match:
                    # 如果同时包含true和false，可以基于它们的位置或上下文做决定
                    # 这里简单地取第一个出现的
                    if message_content.lower().find('true') < message_content.lower().find('false'):
                        return "true"
                    else:
                        return "false"
                else:
                    logger.warning(f"回答既不是true也不是false: {message_content}")
                    return "false"
                    
            except Exception as e:
                logger.error(f"模型调用失败 (尝试 {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error("达到最大重试次数，返回空结果")
                    return "false"