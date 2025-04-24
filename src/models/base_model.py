from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List

class BaseModel(ABC):
    def __init__(self, model_params: Dict[str, Any] = None):
       self.model_params = model_params or {}
       self.max_retries = model_params.get("max_retries", 3)
       self.retry_delay = model_params.get("retry_delay", 2)
            
    @abstractmethod
    def generate_parsing_response(self, prompt, max_retries=3, retry_delay=2):
        pass
    
    @abstractmethod
    def generate_verification_response(self, prompt, text):
        pass
    
    
    def get_params(self, **kwargs) -> Dict[str, Any]:
        """
        获取合并后的参数
        
        Args:
            **kwargs: 方法调用时传入的参数
            
        Returns:
            合并后的参数
        """
        # 优先使用方法调用时传入的参数，其次是初始化时的配置，最后是默认值
        return {**self.params, **kwargs}