from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List

class BaseModel(ABC):
    def __init__(self, params: Dict[str, Any] = None):
       self.params = params or {}
    #    self.max_retries = model_params.get("max_retries", 3)
    #    self.retry_delay = model_params.get("retry_delay", 2)

    @abstractmethod
    def invoke(self, messages:List[Dict[str, str]], **kwargs):
        """
        使用消息格式生成响应
        
        参数:
        - messages: 消息列表，格式为 [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        - **kwargs: 其他参数
        
        返回:
        - 生成的响应
        """
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
    

# if __name__ == "__main__":
#     params = {
#         "temperature": 0.5,
#         "max_new_tokens": 2048,
#         "top_p": 0.9,
#         "do_sample": True,
#         "num_return_sequences": 1
#     }
#     baseModel = BaseModel(params)
#     print(baseModel.get_params())