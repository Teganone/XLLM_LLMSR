from .parsing_generator import ParsingGenerator
from utils.json_utils import JsonUtils
from tqdm import tqdm

class FTParser(ParsingGenerator):
    def __init__(self, task="combined", model="o3-mini"):
        super().__init__(task, model)

    def prompt_extract_combined(self, test_data):
        pass
    
    def prompt_extract_qp(self, test_data):
        pass

    def prompt_extract_cp(self, test_data):
        pass

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
        preprocessed_qp = test_data.get('preprocessed_qp', [])
        preprocessed_cp = test_data.get('preprocessed_cp', [])
        
        user_prompt = self.prompt_template.format(
            question=question,
            cot=cot,
            preprocessed_qp=JsonUtils.format_json(preprocessed_qp),
            preprocessed_cp=JsonUtils.format_json(preprocessed_cp)
        )
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
    
    def _generate_with_messages(self, messages, **kwargs):
        """
        使用消息格式生成响应
        
        参数:
        - messages: 消息列表，格式为 [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        - **kwargs: 其他参数，会传递给模型
        
        返回:
        - 生成的响应
        """
        # 检查模型是否有 _invoke_chat_completions 方法
        if hasattr(self.model, '_invoke_chat_completions') and callable(getattr(self.model, '_invoke_chat_completions')):
            # 直接使用模型的 _invoke_chat_completions 方法
            system_prompt = messages[0]["content"]
            user_prompt = messages[1]["content"]
            response = self.model._invoke_chat_completions(user_prompt, verify=False, system_prompt=system_prompt, **kwargs)
            return response
        else:
            # 如果模型没有 _invoke_chat_completions 方法，则使用 generate_parsing_response 方法
            # 将消息格式转换为单个提示
            full_prompt = f"{messages[0]['content']}\n\n{messages[1]['content']}"
            response = self.model.generate_parsing_response(full_prompt, **kwargs)
            return response
    
    def parse(self, data, output_file, batch_size=10, **kwargs):
        results = []
        
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            
            for item in tqdm(batch_data, desc=f"解析批次 {i//batch_size + 1}/{(len(data)-1)//batch_size + 1}"):
                user_prompt = self.prompt_creator[self.task_name](item)
                
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                # 调用模型生成响应
                response = self._generate_with_messages(messages, **kwargs)
                
                # 处理响应
                try:
                    # 确保response是字典
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
                if len(results) % batch_size == 0:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    print(f"已保存{len(results)}个结果到{output_file}")
        
        # 保存结果到文件
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"解析结果已保存到 {output_file}")
        
        return results
    

    
    