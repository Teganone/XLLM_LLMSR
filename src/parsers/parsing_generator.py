from src.models.llama import LlamaModel
from src.models.openai_model import OpenaiModel
from src.models.base_model import BaseModel
import json
from abc import ABC, abstractmethod
from tqdm import tqdm
from src.utils.json_utils import JsonUtils

class ParsingGenerator(ABC):
    def __init__(self, task="combined", model=None):
        self.task_type = task
        self.prompt_creator = {
            "combined": self.prompt_extract_combined,
            "qp": self.prompt_extract_qp,
            "cp": self.prompt_extract_cp,
        }
    
        self.load_prompt_templates()
        # self._load_model(model)
        self.model = model if model is not None else OpenaiModel("o3-mini")




    # def _load_model(self, model):
    #     '''加载不同的model'''
    #     # 根据模型名称选择合适的模型类
    #     if model in ['o3-mini', 'o4-mini', 'gpt-4o', 'o3', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4', 'o1']:
    #         # OpenAI模型
    #         self.model = OpenaiModel(model=model)
    #     else:
    #         # 默认使用Llama模型
    #         self.model = LlamaModel(model_path=model)

        
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
      
    @abstractmethod
    def set_messages(self, user_prompt):
        pass


    def parse(self, data, output_file, batch_size=10, **kwargs):
        results = []
        for item in tqdm(data, desc="处理测试数据"):
            user_prompt = self.prompt_creator[self.task_type](item)
            # messages = [
            #     {"role": "user", "content": user_prompt}
            # ]
            messages = self.set_messages(user_prompt)
            response = self.model.invoke(messages,**kwargs)
            
            if isinstance(response, str):
                try:
                    response = JsonUtils.extract_json_from_text(response)
                    if not isinstance(response, dict):
                        print(f"提取的内容不是有效的JSON对象: {response[:100]}...")
                        if self.task_type == "qp":
                            response = {"question_parsing": []}
                        elif self.task_type == "cp":
                            response = {"cot_parsing": []}
                        else:  # combined
                            response = {"question_parsing": [], "cot_parsing": []}
                except:
                    print(f"无法解析响应为JSON: {response[:100]}...")
                    if self.task_type == "qp":
                        response = {"question_parsing": []}
                    elif self.task_type == "cp":
                        response = {"cot_parsing": []}
                    else:  # combined
                        response = {"question_parsing": [], "cot_parsing": []}
                
            if self.task_type == "qp" or self.task_type == "combined":
                if "question_parsing" not in response:
                    response["question_parsing"] = []
            
            if self.task_type == "cp" or self.task_type == "combined":
                if "cot_parsing" not in response:
                    response["cot_parsing"] = []
                
            result = {
                "question": item['question'],
                "answer": item["answer"],
                "id": item["id"],
                "cot": item["cot"],
            }

            if self.task_type == "qp" and 'cot_parsing' in item.keys():
                result["cot_parsing"] = item['cot_parsing']
            if self.task_type == "cp" and 'question_parsing' in item.keys():
                result["question_parsing"] = item['question_parsing']

            # 根据任务类型添加解析结果
            if self.task_type == "qp" or self.task_type == "combined":
                result["question_parsing"] = response["question_parsing"]
            
            if self.task_type == "cp" or self.task_type == "combined":
                result["cot_parsing"] = response["cot_parsing"]
            
            if 'sel_idx' in item.keys():
                result['sel_idx'] = item['sel_idx']
                
            results.append(result)
            
            # 每处理batch_size个样本，保存一次结果
            if len(results) % batch_size == 0:
                JsonUtils.save_to_file(results, output_file)
                print(f"已保存{len(results)}个结果到{output_file}")

        JsonUtils.save_to_file(results, output_file)
        print(f"处理完成，共{len(results)}个结果已保存到{output_file}")

        
