import json
import time
import re
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# MODEL = "meta-llama/Llama-3-8B-Instruct"
# MODEL = '/datacenter/models/LLM-Research/Llama-3-8B-Instruct'
# RESPONSE_TEMPERATURE = 0.5
def extract_json_from_text(text):
    """
    从文本中提取JSON部分
    
    Args:
        text: 包含JSON的文本
    
    Returns:
        提取的JSON对象或原始文本（如果提取失败）
    """
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 尝试查找JSON对象的开始和结束
    try:
        # 查找第一个左大括号和最后一个右大括号
        start = text.find('{')
        end = text.rfind('}') + 1
        
        if start >= 0 and end > start:
            json_str = text[start:end]
            return json.loads(json_str)
    except:
        pass
    
    # 尝试使用正则表达式查找JSON对象
    try:
        json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
        matches = re.findall(json_pattern, text)
        if matches:
            for potential_json in matches:
                try:
                    return json.loads(potential_json)
                except:
                    continue
    except:
        pass
    
    return text

def get_validation_response(prompt, text, model_path, pipe=None, tokenizer=None, temperature=1, max_retries=3, retry_delay=2):
    """
    使用Llama-3-8B-Instruct模型获取响应
    
    Args:
        prompt: 系统提示
        text: 用户输入
        pipe: 预加载的模型pipeline（如果提供）
        tokenizer: 预加载的tokenizer（如果提供）
        model_path: 模型路径（如果pipe和tokenizer未提供）
        temperature: 温度参数
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
    
    Returns:
        模型响应内容
    """
    for attempt in range(max_retries):
        try:
            # 如果没有提供预加载的模型和分词器，则加载
            if pipe is None or tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                # 根据可用资源决定是否使用半精度
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"使用设备: {device}")
                
                # 使用pipeline简化模型加载和推理过程
                pipe = pipeline(
                    "text-generation",
                    model=model_path,
                    tokenizer=tokenizer,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto",
                )
            
            # 构建Llama格式的提示
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ]
            
            # 将消息格式化为Llama模型期望的格式
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False)
            
            # 生成响应
            outputs = pipe(
                prompt_text,
                max_new_tokens=1024,
                temperature=temperature,
                do_sample=False,
                top_p=0.8,
                num_return_sequences=1,
            )
            
            # 提取生成的文本
            generated_text = outputs[0]["generated_text"]
            
            # 提取模型回复（去除输入提示）
            response = generated_text[len(prompt_text):].strip()
            
            return response
                
        except Exception as e:
            print(f"模型调用失败 (尝试 {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print("达到最大重试次数，返回空结果")
                return {}


def get_response(prompt, model_path, pipe=None, tokenizer=None, temperature=0.5, max_retries=3, retry_delay=2):
    """
    使用Llama-3-8B-Instruct模型获取响应
    
    Args:
        prompt: 用户提示
        pipe: 预加载的模型pipeline（如果提供）
        tokenizer: 预加载的tokenizer（如果提供）
        model_path: 模型路径（如果pipe和tokenizer未提供）
        temperature: 温度参数
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
    
    Returns:
        模型响应内容
    """
    for attempt in range(max_retries):
        try:
            # 如果没有提供预加载的模型和分词器，则加载
            if pipe is None or tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                # 根据可用资源决定是否使用半精度
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"使用设备: {device}")
                
                # 使用pipeline简化模型加载和推理过程
                pipe = pipeline(
                    "text-generation",
                    model=model_path,
                    tokenizer=tokenizer,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto",
                )
            
            # 构建Llama格式的提示
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # 将消息格式化为Llama模型期望的格式
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False)
            
            # 生成响应
            outputs = pipe(
                prompt_text,
                max_new_tokens=1024,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                num_return_sequences=1,
            )
            
            # 提取生成的文本
            generated_text = outputs[0]["generated_text"]
            
            # 提取模型回复（去除输入提示）
            response = generated_text[len(prompt_text):].strip()
            
            # 尝试提取和解析JSON
            try:
                json_content = extract_json_from_text(response)
                if isinstance(json_content, dict):
                    return json_content
                else:
                    print(f"无法提取有效JSON: {response[:100]}...")
                    return response
            except Exception as e:
                print(f"JSON处理失败: {str(e)}")
                return response
                
        except Exception as e:
            print(f"模型调用失败 (尝试 {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print("达到最大重试次数，返回空结果")
                return {}


class ParsingGenerator:
    def __init__(self, model_path, task_type="combined", version='v5',temperature=0.5):
        self.version = version
        self.task_type = task_type  # 新增：任务类型
        self.model_path = model_path
        self.temperature = temperature
        # 根据任务类型设置任务名称
        if task_type == "qp":
            self.task_name = f"extract_qp_{self.version}"
        elif task_type == "cp":
            self.task_name = f"extract_cp_{self.version}"
        else:  # combined
            self.task_name = f"extract_{self.version}"
        
        # 设置 prompt 创建函数
        self.prompt_creator = {
            f"extract_{self.version}": self.prompt_extract_combined,
            f"extract_qp_{self.version}": self.prompt_extract_qp,
            f"extract_cp_{self.version}": self.prompt_extract_cp,
        }
        
        self.load_prompt_templates()
    
        # 加载模型和分词器（只加载一次）
        print(f"正在加载模型 {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 根据可用资源决定是否使用半精度
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        
        # 使用pipeline简化模型加载和推理过程
        self.pipe = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
        )
        print("模型加载完成！")

    def load_prompt_templates(self):
        """加载不同任务类型的 prompt 模板"""
        if self.task_type == "qp":
            prompt_file = f'prompts/extract_qp_{self.version}.txt'
        elif self.task_type == "cp":
            prompt_file = f'prompts/extract_cp_{self.version}.txt'
        else:  # combined
            prompt_file = f'prompts/extract_{self.version}.txt'
            
        with open(prompt_file, 'r') as f:
            self.prompt_template = f.read()

    def prompt_extract_combined(self, test_data):
        """组合任务的 prompt 创建函数"""
        question = test_data['question']
        cot = test_data['cot']
        full_prompt = self.prompt_template.replace('[[question]]', question).replace('[[cot]]', cot)
        full_prompt += "\n\n请只返回有效的JSON格式，不要添加任何额外的文本、解释或注释。输出应该是一个包含'question_parsing'和'cot_parsing'两个键的JSON对象。"
        # full_prompt += "\n\nPlease only return valid JSON format, without any extra text, explanations, or annotations. The output should be a JSON object containing two keys: 'question_parsing' and 'cot_parsing'."
        return full_prompt
        
    def prompt_extract_qp(self, test_data):
        """问题解析任务的 prompt 创建函数"""
        question = test_data['question']
        full_prompt = self.prompt_template.replace('[[question]]', question)
        full_prompt += "\n\n请只返回有效的JSON格式，不要添加任何额外的文本、解释或注释。输出应该是一个包含'question_parsing'键的JSON对象。"
        # full_prompt += "\n\nPlease only return valid JSON format, without any extra text, explanations, or annotations. The output should be a JSON object containing one key: 'question_parsing'."
        return full_prompt
        
    def prompt_extract_cp(self, test_data):
        """思维链解析任务的 prompt 创建函数"""
        question = test_data['question']
        cot = test_data['cot']
        
        question_parsing_str = json.dumps(test_data['question_parsing'], ensure_ascii=False, indent=2)
        full_prompt = self.prompt_template.replace('[[question]]', question).replace('[[cot]]', cot).replace('[[question_parsing]]',question_parsing_str)
        
        # 如果有question_parsing，也添加到提示中
        # if 'question_parsing' in test_data:
        #     question_parsing_str = json.dumps(test_data['question_parsing'], ensure_ascii=False, indent=2)
        #     full_prompt.replace('[[question_parsing]]',question_parsing_str)
        full_prompt += "\n\n请只返回有效的JSON格式，不要添加任何额外的文本、解释或注释。输出应该是一个包含'cot_parsing'键的JSON对象。"
        # full_prompt += "\n\nPlease only return valid JSON format, without any extra text, explanations, or annotations. The output should be a JSON object containing one key: 'cot_parsing'."
        return full_prompt

    def load_dataset():
        pass
    
    def process_test_data(self, test_data, output_file, batch_size=10):
        results = []
        
        # 处理每个测试样本
        for item in tqdm(test_data, desc="处理测试数据"):
            full_prompt = self.prompt_creator[self.task_name](item) 
        
            response = get_response(full_prompt, model_path=self.model_path, pipe = self.pipe, tokenizer=self.tokenizer, temperature=self.temperature)
            
            # 确保response是字典
            if isinstance(response, str):
                try:
                    # 尝试使用提取函数
                    response = extract_json_from_text(response)
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
            
                
            # 确保包含所需字段
            if self.task_type == "qp" or self.task_type == "combined":
                if "question_parsing" not in response:
                    response["question_parsing"] = []
            
            if self.task_type == "cp" or self.task_type == "combined":
                if "cot_parsing" not in response:
                    response["cot_parsing"] = []
                
                
            # 构建结果
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
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"已保存{len(results)}个结果到{output_file}")
        
        # 保存最终结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"处理完成，共{len(results)}个结果已保存到{output_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="使用 Llama 模型处理数据")
    parser.add_argument("--task_type", type=str, choices=["qp", "cp", "combined"], 
                        default="combined", help="任务类型：问题解析(qp)、思维链解析(cp)或组合任务(combined)")
    parser.add_argument("--input_file", type=str, 
                        # default='LLMSR_Datasets/Public_Test_A.json',
                        required=True,
                        # default='LLMSR_Datasets/Output Results/Test_A_result_qp_v5_0.2_Llama-3-8B-Instruct-icl.json',
                        help="输入文件路径")
    parser.add_argument("--output_file", type=str, 
                        required=True, help="输出文件路径，如果不指定则自动生成")
    parser.add_argument("--temperature", type=float, 
                        default=0.5, help="生成温度参数")
    parser.add_argument("--version", type=str, 
                        default='v5', help="prompt version")
    parser.add_argument("--batch_size", type=int, 
                        default=5, help="批处理大小")
    parser.add_argument("--model_path", type=str, default='/datacenter/yingjiahao/models/Llama-3-8B-Instruct', help='模型路径')
    args = parser.parse_args()
    
    # 如果没有指定输出文件，则自动生成
    if args.output_file is None:
        task_suffix = f"_{args.task_type}" if args.task_type != "combined" else ""
        output_file = f'LLMSR_Datasets/Output Results/Test_A_result{task_suffix}_{args.version}_{args.temperature}_{args.model_path.split("/")[-1]}-icl.json'
    else:
        output_file = args.output_file
    
    # 创建 ParsingGenerator 实例
    parsing_Generator = ParsingGenerator(model_path=args.model_path, task_type=args.task_type, version=args.version, temperature=args.temperature)
    
    # 加载数据
    with open(args.input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # 处理数据
    parsing_Generator.process_test_data(
        test_data=raw_data,
        output_file=output_file,
        batch_size=args.batch_size
    )
