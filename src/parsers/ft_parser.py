from src.parsers.parsing_generator import ParsingGenerator
from src.utils.json_utils import JsonUtils
from tqdm import tqdm

class FTParser(ParsingGenerator):
    def __init__(self, task="combined", model="o3-mini"):
        super().__init__(task, model)

    def load_prompt_templates(self):
        """加载提示模板"""
        prompt_file = 'prompts/parser_ft.txt'
        try:
            with open(prompt_file, 'r') as f:
                content = f.read()
                
                # 分割系统提示和用户提示
                parts = content.split('# USER_PROMPT')
                if len(parts) > 1:
                    system_part = parts[0].replace('# SYSTEM_PROMPT', '').strip()
                    user_part = parts[1].strip()
                    
                    self.system_prompt = system_part
                    self.prompt_template = user_part
                else:
                    # 如果没有明确分割，则使用整个内容作为用户提示
                    self.system_prompt = "You are an expert in logical parsing analysis."
                    self.prompt_template = content
                    
        except FileNotFoundError:
            print(f"提示模板文件 {prompt_file} 不存在，使用默认模板")
            self.system_prompt = "You are an expert in logical parsing and reasoning analysis, specializing in analyzing problem conditions and chain-of-thought reasoning processes."
            self.prompt_template = self._get_default_prompt_template()


    
    def prompt_extract_combined(self, test_data):
        """组合任务的 prompt 创建函数"""
        question = test_data.get('question', '')
        cot = test_data.get('cot', '')
        # preprocessed_qp = test_data.get('preprocessed_qp', [])
        # preprocessed_cp = test_data.get('preprocessed_cp', [])
        preprocessed_qp = test_data.get('preprocessed_qp') or test_data.get('question_parsing', [])
        preprocessed_cp = test_data.get('preprocessed_cp') or test_data.get('cot_parsing', [])
        
        user_prompt = self.prompt_template.format(
            question=question,
            cot=cot,
            preprocessed_qp=JsonUtils.format_json(preprocessed_qp),
            preprocessed_cp=JsonUtils.format_json(preprocessed_cp)
        )
        print(user_prompt)
        return user_prompt
    
    def prompt_extract_qp(self, test_data):
        """问题解析任务的 prompt 创建函数"""
        question = test_data.get('question', '')
        preprocessed_qp = test_data.get('preprocessed_qp', [])
        
        user_prompt = self.prompt_template.format(
            question=question,
            cot="",
            preprocessed_qp=JsonUtils.format_json(preprocessed_qp),
            preprocessed_cp="[]"
        )
        
        return user_prompt
    
    def prompt_extract_cp(self, test_data):
        """思维链解析任务的 prompt 创建函数"""
        question = test_data.get('question', '')
        cot = test_data.get('cot', '')
        preprocessed_cp = test_data.get('preprocessed_cp', [])
        question_parsing = test_data.get('question_parsing', [])
        
        user_prompt = self.prompt_template.format(
            question=question,
            cot=cot,
            preprocessed_qp=JsonUtils.format_json(question_parsing),
            preprocessed_cp=JsonUtils.format_json(preprocessed_cp)
        )
        
        return user_prompt
    
    def set_messages(self, user_prompt):
        messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
        return messages

    def parse(self, data, output_file, batch_size=10, **kwargs):
        results = []
        
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            
            for item in tqdm(batch_data, desc=f"解析批次 {i//batch_size + 1}/{(len(data)-1)//batch_size + 1}"):
                user_prompt = self.prompt_creator[self.task_type](item)
                
                messages = self.set_messages(user_prompt)
                
                # 调用模型生成响应
                response = self.model.invoke(messages, **kwargs)
                # print("response:",response)
                # 处理响应
                try:
                    if isinstance(response, str):
                        parsed_prediction = JsonUtils.extract_json_from_text(response)
                        if not isinstance(parsed_prediction, dict):
                            print(f"提取的内容不是有效的JSON对象: {response[:100]}...")
                            if self.task_type == "qp":
                                parsed_prediction = {"question_parsing": []}
                            elif self.task_type == "cp":
                                parsed_prediction = {"cot_parsing": []}
                            else:  # combined
                                parsed_prediction = {"question_parsing": [], "cot_parsing": []}
                    else:
                        parsed_prediction = response
                    
                    # 更新结果
                    result = {
                        "id": item.get('id', ''),
                        "question": item.get('question', ''),
                        "cot": item.get('cot', ''),
                        "answer": item.get('answer', '')
                    }
                    
                    # 根据任务类型添加解析结果
                    if self.task_type == "qp" or self.task_type == "combined":
                        result["question_parsing"] = parsed_prediction.get("question_parsing", [])
                    
                    if self.task_type == "cp" or self.task_type == "combined":
                        # 确保cot_parsing中的每个项目都有必要的字段
                        cot_parsing = parsed_prediction.get("cot_parsing", [])
                        validated_cot_parsing = []
                        for entry in cot_parsing:
                            if isinstance(entry, dict) and "statement" in entry and "evidence" in entry:
                                if "Verification" not in entry:
                                    entry["Verification"] = "true"  # 默认值
                                validated_cot_parsing.append(entry)
                        result["cot_parsing"] = validated_cot_parsing
                    
                    # 保留原始数据中的其他字段
                    if self.task_type == "qp" and 'cot_parsing' in item:
                        result["cot_parsing"] = item['cot_parsing']
                    if self.task_type == "cp" and 'question_parsing' in item:
                        result["question_parsing"] = item['question_parsing']
                    if 'sel_idx' in item:
                        result['sel_idx'] = item['sel_idx']
                    
                    results.append(result)
                except Exception as e:
                    print(f"解析预测结果失败: {e}")
                    print(f"原始输出: {response}")
                    # 如果解析失败，添加空结果
                    result = {
                        "id": item.get('id', ''),
                        "question": item.get('question', ''),
                        "cot": item.get('cot', ''),
                        "answer": item.get('answer', '')
                    }
                    
                    if self.task_type == "qp" or self.task_type == "combined":
                        result["question_parsing"] = []
                    
                    if self.task_type == "cp" or self.task_type == "combined":
                        result["cot_parsing"] = []
                        
                    results.append(result)
                
                # 每处理batch_size个样本，保存一次结果
                if output_file and len(results) % batch_size == 0:
                    JsonUtils.save_to_file(results, output_file)
                    print(f"已保存{len(results)}个结果到{output_file}")
        
        # 保存结果到文件
        if output_file:
            JsonUtils.save_to_file(results, output_file)
            print(f"解析结果已保存到 {output_file}")
        
        return results
    
if __name__ == '__main__':  
    from src.models.openai_model import OpenaiModel
    from src.models.llama import LlamaModel
    from src.verifiers.verifier_factory import VerifierFactory
    import json
    # model = LlamaModel(model_path="/datacenter/models/LLM-Research/Llama-3-8B-Instruct")
    model = LlamaModel(model_path='/datacenter/chendanchun/models/finetune/Llama-3-8B-Instruct_o3-mini-high_combined/final_model')
    parser = FTParser(model=model)
    # parser = ICLParser(model="/datacenter/models/LLM-Research/Llama-3-8B-Instruct")
    data = JsonUtils.load_json('results/78_54_23_15.json')[:3]
    parser.parse(data,output_file='results/test_llama.log')

    
    # model = OpenaiModel("o3-mini",{"reasoning_effect":'low'})
    # parser = FTParser(model=model)
    # data = JsonUtils.load_json('data/Public_Test_A.json')[:3]
    # parser.parse(data,output_file='results/openai_test.log')
    