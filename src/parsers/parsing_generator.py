from models.llama import LlamaModel
from models.openai_model import OpenaiModel
from models.base_model import BaseModel
import json
from abc import ABC, abstractmethod

class ParsingGenerator(ABC):
    def __init__(self, task="combined", model="o3-mini"):
        self.task_type = task
        self.prompt_creator = {
            "combined": self.prompt_extract_combined,
            "qp": self.prompt_extract_qp,
            "cp": self.prompt_extract_cp,
        }
        self.load_prompt_templates()

        self._load_model(model)

    def _load_model(self,model):
        '''加载不同的model'''
        #可以用模型工厂来选择openai和llama
        # self.model = 
        
    @abstractmethod
    def load_prompt_templates(self):
        """加载不同任务类型的 prompt 模板"""
        pass
    
    @abstractmethod
    def prompt_extract_combined(self, test_data):
        """组合任务的 prompt 创建函数"""
    
    @abstractmethod
    def prompt_extract_qp(self, test_data):
        """问题解析任务的 prompt 创建函数"""
       
    
    @abstractmethod
    def prompt_extract_cp(self, test_data):
        """思维链解析任务的 prompt 创建函数"""
      
    
        
    # def _load_model(self, model):
    #     if model in ['o3-mini','o4-mini','gpt-4o','o3','gpt-4.1','gpt-4.1-mini','gpt-4','o1']:
            
    @abstractmethod
    def parse(self, data, output_file, batch_size=10, **kwargs):
        pass
        
