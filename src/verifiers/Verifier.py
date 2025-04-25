from src.models.openai_model import OpenaiModel
from abc import ABC, abstractmethod

class Verifier(ABC):
    def __init__(self, model=None):
        """
        初始化验证器
        
        参数:
        - model: 用于验证的模型
        """
        self.model = model if model is not None else OpenaiModel("o3-mini")
        # self.prompt_creator = {
        #     "llm": self.
        #     "z3":
        # }
        self.load_prompt_templates()
        

    def _load_prompt(self, method):
        # "llm": LLM_VERIFT_SYSTEM_PROMPT
        # "z3": 
        pass

    @abstractmethod
    def load_prompt_templates(self):
        pass


    @abstractmethod
    def verify(self, data, output_file, batch_size=10, **kwargs):
        """
        验证陈述是否可以从证据中推导出来
        
        参数:
        - statement: 陈述
        - evidence: 证据
        - **kwargs: 其他参数
        
        返回:
        - 验证结果
        """
        pass



    
    

    