from src.parsers.parsing_generator import ParsingGenerator
from src.utils.json_utils import JsonUtils
from src.utils.logging_utils import LoggingUtils

# 设置日志
logger = LoggingUtils.setup_logger(
    name="icl_parser",
    # log_file="logs/iclparser.log"
    )

class ICLParser(ParsingGenerator):
    def __init__(self, task="combined", model="o3-mini"):
        super().__init__(task, model)
    
    def load_prompt_templates(self):
        prompt_file = f'prompts/extract_{self.task_type}.txt'
        try:
            with open(prompt_file, 'r') as f:
                self.prompt_template = f.read()
        except FileNotFoundError:
            print(f"提示模板文件 {prompt_file} 不存在")
            raise

    def prompt_extract_combined(self, test_data):
        """组合任务的 prompt 创建函数"""
        question = test_data['question']
        cot = test_data['cot']
        full_prompt = self.prompt_template.replace('[[question]]', question).replace('[[cot]]', cot)
        full_prompt += "\n\n请只返回有效的JSON格式，不要添加任何额外的文本、解释或注释。输出应该是一个包含'question_parsing'和'cot_parsing'两个键的JSON对象。"
        return full_prompt
    
    def prompt_extract_qp(self, test_data):
        """问题解析任务的 prompt 创建函数"""
        question = test_data['question']
        full_prompt = self.prompt_template.replace('[[question]]', question)
        full_prompt += "\n\n请只返回有效的JSON格式，不要添加任何额外的文本、解释或注释。输出应该是一个包含'question_parsing'键的JSON对象。"
        return full_prompt
    
    def prompt_extract_cp(self, test_data):
        """思维链解析任务的 prompt 创建函数"""
        question = test_data['question']
        cot = test_data['cot']
        
        question_parsing_str = JsonUtils.format_json(test_data['question_parsing'])
        full_prompt = self.prompt_template.replace('[[question]]', question).replace('[[cot]]', cot).replace('[[question_parsing]]',question_parsing_str)
        full_prompt += "\n\n请只返回有效的JSON格式，不要添加任何额外的文本、解释或注释。输出应该是一个包含'cot_parsing'键的JSON对象。"
        return full_prompt
    
    def set_messages(self, user_prompt):
        messages = [
                {"role": "user", "content": user_prompt}
            ]
        return messages

    # def parse(self, data, output_file, batch_size=10, **kwargs):
    #     results = []
    #     for item in tqdm(data, desc="处理测试数据"):
    #         user_prompt = self.prompt_creator[self.task_type](item)
    #         # messages = [
    #         #     {"role": "user", "content": user_prompt}
    #         # ]
    #         messages = self.set_messages(user_prompt)
    #         response = self.model.invoke(messages,**kwargs)
            
    #         if isinstance(response, str):
    #             try:
    #                 response = JsonUtils.extract_json_from_text(response)
    #                 if not isinstance(response, dict):
    #                     print(f"提取的内容不是有效的JSON对象: {response[:100]}...")
    #                     if self.task_type == "qp":
    #                         response = {"question_parsing": []}
    #                     elif self.task_type == "cp":
    #                         response = {"cot_parsing": []}
    #                     else:  # combined
    #                         response = {"question_parsing": [], "cot_parsing": []}
    #             except:
    #                 print(f"无法解析响应为JSON: {response[:100]}...")
    #                 if self.task_type == "qp":
    #                     response = {"question_parsing": []}
    #                 elif self.task_type == "cp":
    #                     response = {"cot_parsing": []}
    #                 else:  # combined
    #                     response = {"question_parsing": [], "cot_parsing": []}
                
    #         if self.task_type == "qp" or self.task_type == "combined":
    #             if "question_parsing" not in response:
    #                 response["question_parsing"] = []
            
    #         if self.task_type == "cp" or self.task_type == "combined":
    #             if "cot_parsing" not in response:
    #                 response["cot_parsing"] = []
                
    #         result = {
    #             "question": item['question'],
    #             "answer": item["answer"],
    #             "id": item["id"],
    #             "cot": item["cot"],
    #         }

    #         if self.task_type == "qp" and 'cot_parsing' in item.keys():
    #             result["cot_parsing"] = item['cot_parsing']
    #         if self.task_type == "cp" and 'question_parsing' in item.keys():
    #             result["question_parsing"] = item['question_parsing']

    #         # 根据任务类型添加解析结果
    #         if self.task_type == "qp" or self.task_type == "combined":
    #             result["question_parsing"] = response["question_parsing"]
            
    #         if self.task_type == "cp" or self.task_type == "combined":
    #             result["cot_parsing"] = response["cot_parsing"]
            
    #         if 'sel_idx' in item.keys():
    #             result['sel_idx'] = item['sel_idx']
                
    #         results.append(result)
            
    #         # 每处理batch_size个样本，保存一次结果
    #         if len(results) % batch_size == 0:
    #             JsonUtils.save_to_file(results, output_file)
    #             print(f"已保存{len(results)}个结果到{output_file}")

    #     JsonUtils.save_to_file(results, output_file)
    #     print(f"处理完成，共{len(results)}个结果已保存到{output_file}")


if __name__ == '__main__':
    parser = ICLParser(model='gpt-4')
    data = JsonUtils.load_json('/Users/I757479/Documents/biye/XLLM_LLMSR/data/Public_Test_A.json')[:3]
    parser.parse(data,output_file='results/test.log')