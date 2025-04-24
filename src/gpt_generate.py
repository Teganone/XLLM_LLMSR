import os
import sys
import json
import time
from tqdm import tqdm
from openai import AzureOpenAI
from dotenv import load_dotenv
import argparse
from src.z3_solver.smt_solver import LLMSR_Z3_Program

load_dotenv(dotenv_path='.env')

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ['AZURE_OPENAI_API_KEY'],
    api_version=os.environ['OPENAI_API_VERSION'],
)

MODEL = "o3-mini"


def get_validation_response(prompt, text, model="o3-mini", temperature=1, max_retries=3, retry_delay=2):
    """
    使用Azure OpenAI API获取响应
    
    Args:
        prompt: 系统提示
        text: 用户输入
        model: 模型名称
        temperature: 温度参数
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
    
    Returns:
        模型响应内容
    """
    for attempt in range(max_retries):
        try:
            reasoning_params = (
                {"reasoning_effort": "high"}
                if model in {"o1", "o3-mini"}
                else {"top_p": 0.5, "temperature": 0.9}
            )
            completion = client.chat.completions.create(
                model=model,
                # temperature=temperature,
                **reasoning_params,
                messages=[
                    # {"role": "user", "content": prompt},
                    {"role": "system", "content": f"{prompt}" },
                    {"role": "user", "content": f"{text}"},
                    
                ],
                # response_format={"type": "json_object"}
            )
            
            message = completion.choices[0].message
            if message.content:
                return message.content
            else:
                print("模型返回空内容")
                return {}
                
        except Exception as e:
            print(f"API调用失败 (尝试 {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print("达到最大重试次数，返回空结果")
                return {}


def get_response(prompt, model="o3-mini", temperature=1, max_retries=3, retry_delay=2):
    """
    使用Azure OpenAI API获取响应
    
    Args:
        prompt: 系统提示
        text: 用户输入
        model: 模型名称
        temperature: 温度参数
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
    
    Returns:
        模型响应内容
    """
    for attempt in range(max_retries):
        try:
            reasoning_params = (
                {"reasoning_effort": "high"}
                if model in {"o1", "o3-mini"}
                else {"top_p": 0.9, "temperature": 0.9}
            )

            completion = client.chat.completions.create(
                model=model,
                **reasoning_params,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                # response_format={"type": "json_object"}
            )
            
            message = completion.choices[0].message
            if message.content:
                try:
                    # 尝试解析JSON
                    json_content = json.loads(message.content)
                    return json_content
                except json.JSONDecodeError:
                    print(f"JSON解析失败，尝试修复格式: {message.content[:100]}...")
                    # 返回原始内容，后续可以尝试修复
                    return message.content
            else:
                print("模型返回空内容")
                return {}
                
        except Exception as e:
            print(f"API调用失败 (尝试 {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print("达到最大重试次数，返回空结果")
                return {}


def get_logic_program_response(prompt, model="o3-mini", temperature=1, max_retries=3, retry_delay=2):
    """
    使用Azure OpenAI API获取逻辑程序响应
    
    Args:
        prompt: 提示文本
        model: 模型名称
        temperature: 温度参数
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
    
    Returns:
        生成的逻辑程序
    """
    for attempt in range(max_retries):
        try:
            reasoning_params = (
                {"reasoning_effort": "high"}
                if model in {"o1", "o3-mini"}
                else {"top_p": 0.5, "temperature": 0.9}
            )
            completion = client.chat.completions.create(
                model=model,
                # temperature=temperature,
                **reasoning_params,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            
            message = completion.choices[0].message
            if message.content:
                # 尝试提取逻辑程序部分
                content = message.content
                if "###" in content:
                    logic_program = content.split("###")[1].strip()
                elif "# Declarations" in content:
                    logic_program = content[content.find("# Declarations"):].strip()
                else:
                    logic_program = content
                return logic_program
            else:
                print("模型返回空内容")
                return ""
                
        except Exception as e:
            print(f"API调用失败 (尝试 {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print("达到最大重试次数，返回空结果")
                return ""


def extract_verification_results(output):
    """
    从输出中提取验证结果列表
    
    Args:
        output: 执行逻辑程序的输出
        
    Returns:
        验证结果列表（布尔值）
    """
    # 查找包含"All verification results:"的行
    for line in output:
        if "All verification results:" in line:
            import re
            match = re.search(r'\[(.*?)\]', line)
            if match:
                # 提取列表内容并分割
                results_str = match.group(1)
                verification_results = [r.strip() for r in results_str.split(', ')]
                verification_results = [result.lower() for result in verification_results]
                return verification_results

    
    return []  # 如果没有找到结果，返回空列表


class ParsingGenerator:
    def __init__(self, task):
        self.task = task
        # self.version = "v7"
        # self.task_name = f"extract_{self.version}"
        self.prompt_creator = {
            "parsing": self.prompt_parsing,
            "verification": self.prompt_verification,
        }
        self.prompt_file = {
            "parsing": './src/prompts/extract_v7.txt',
            "verification": 'src/prompts/solver_verification_v0.txt',
        }
        self.load_prompt_templates()

    def load_prompt_templates(self):
        with open(self.prompt_file[self.task], 'r') as f:
            self.prompt_template = f.read()

    def prompt_parsing(self, test_data):
        question = test_data['question']
        cot = test_data['cot']
        full_prompt = self.prompt_template.replace('[[question]]', question).replace('[[cot]]', cot)
        return full_prompt

    def prompt_verification(self, test_data):
        question = test_data['question']
        cot_parsing = json.dumps(test_data['cot_parsing'], indent=4, ensure_ascii=False)
        full_prompt = self.prompt_template.replace('[[QUESTION]]', question).replace('[[COT_PARSING]]', cot_parsing)
        return full_prompt

    def execute_logic_program(self, logic_program, debug=True):
        """
        执行逻辑程序并获取验证结果
        
        Args:
            logic_program: 逻辑程序
            debug: 是否打印调试信息
            
        Returns:
            验证结果列表和错误信息
        """
        try:
            if debug:
                print("执行逻辑程序:")
                print("-" * 50)
                print(logic_program)
                print("-" * 50)
            
            z3_program = LLMSR_Z3_Program(logic_program, 'LLMSR')

            output, error_message = z3_program.execute_program()
            
            if debug:
                print("执行结果:")
                print("-" * 50)
                for line in output:
                    print(line)
                print("-" * 50)
                if error_message:
                    print(f"错误信息: {error_message}")
            
            return output, error_message, z3_program.standard_code
        except Exception as e:
            # if debug:
            print(f"执行逻辑程序时出错: {str(e)}")
            return [], str(e), z3_program.standard_code

    def process_test_data(self, test_data, output_file, batch_size=10):
        results = []
        
        # 处理每个测试样本
        for item in tqdm(test_data, desc="处理测试数据"):
            full_prompt = self.prompt_creator[self.task](item) 
            
            if self.task == "parsing":
                # 解析任务，获取JSON响应
                response = get_response(full_prompt)
                
                # 确保response是字典
                if isinstance(response, str):
                    try:
                        response = json.loads(response)
                    except:
                        print(f"无法解析响应为JSON: {response[:100]}...")
                        response = {
                            "question_parsing": [],
                            "cot_parsing": []
                        }
                
                # 确保包含所需字段
                if "question_parsing" not in response:
                    response["question_parsing"] = []
                if "cot_parsing" not in response:
                    response["cot_parsing"] = []
                    
                # 构建结果
                result = {
                    "question": item['question'],
                    "question_parsing": response["question_parsing"],
                    "answer": item["answer"],
                    "id": item["id"],
                    "cot": item["cot"],
                    "cot_parsing": response["cot_parsing"],
                }
                if 'sel_idx' in item.keys():
                    result['sel_idx'] = item['sel_idx']
            
            elif self.task == "verification":
                # 验证任务，获取逻辑程序
                logic_program = get_logic_program_response(full_prompt)
                
                # 迭代尝试，最多3次
                max_retries = 3
                for retry in range(max_retries):
                    # 执行逻辑程序
                    output, error_message, standard_code = self.execute_logic_program(logic_program, debug=(retry==0))
                    
                    # 如果执行成功或已达到最大重试次数，跳出循环
                    if not error_message or retry == max_retries - 1:
                        break
                    
                    # 构建修复提示，包含错误信息和之前的逻辑程序
                    repair_prompt = f"""
我尝试执行以下逻辑程序时遇到错误：

```
{logic_program}
```

错误信息：
{error_message}

请根据错误信息修复这个逻辑程序。确保修复后的程序遵循以下规则：
1. 使用Python语法而不是数学符号（使用and而不是∧，使用or而不是∨，使用not而不是¬）
2. 确保所有变量都已定义
3. 不要对枚举类型使用比较操作符（<, >, <=, >=），除非该函数返回整数类型
4. 使用==进行相等比较，而不是Equals函数
5. 确保所有自定义函数（如Adjacent）都已在声明部分定义

原始问题和COT解析：
{full_prompt}

请提供修复后的完整逻辑程序：
"""
                    print(f"尝试第{retry+1}次修复逻辑程序...")
                    # 获取修复后的逻辑程序
                    logic_program = get_logic_program_response(repair_prompt)
                
                # 更新验证结果
                cot_parsing = item['cot_parsing']
                if not error_message and len(output) > 0:
                    # 提取验证结果
                    verification_results = extract_verification_results(output)
                    print(f"提取的验证结果: {verification_results}")
                    
                    # 更新cot_parsing中的验证结果
                    for i, verification in enumerate(verification_results):
                        if i < len(cot_parsing):
                            cot_parsing[i]['Verification'] = verification
                else:
                    print(f"执行逻辑程序时出错: {error_message}")
                    print("-" * 50)
                    print(item['id'])
                    print("-" * 50)
                
                # 构建结果
                result = item.copy()
                result['cot_parsing'] = cot_parsing
                
                # 可选：保存生成的逻辑程序
                if not os.path.exists('logic_programs'):
                    os.makedirs('logic_programs')
                with open(f"logic_programs/{item['id']}.txt", 'w', encoding='utf-8') as f:
                    f.write(logic_program)
            
                if standard_code is not None:
                    if not os.path.exists('python_programs'):
                        os.makedirs('python_programs')
                    with open(f"python_programs/{item['id']}.py", 'w', encoding='utf-8') as f:
                        f.write(standard_code)
                
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


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input", type=str, default="LLMSR_Datasets/Output Results/Public_Test_result_v5_Llama-3-8B-Instruct-icl.json", help="Input JSON file")
    parser.add_argument("--input", type=str, default="LLMSR_Datasets/Public_Test_A.json", help="Input JSON file")
    # parser.add_argument("--input",type=str,default='LLMSR_Datasets/Final_Selection_Train_v2.json')
    # parser.add_argument("--output", type=str, default=f"LLMSR_Datasets/Output Results/Final_Selection_Train_result_v5_reasoning_{MODEL}-icl.json", help="Output JSON file")

    parser.add_argument("--output", type=str, default=f"LLMSR_Datasets/Output Results/Public_Test_A_result_v7_reasoning_{MODEL}-icl.json", help="Output JSON file")
    # parser.add_argument("--output", type=str, default="LLMSR_Datasets/Output Results/Z3_Verified_Public_Test_result_v5_Llama-3-8B-Instruct-icl_3.json", help="Output JSON file")
    parser.add_argument("--task", type=str, default="parsing", help='parsing (question parsing and cot_parsing) or verification (verification)')
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for saving results")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建ParsingGenerator实例
    parsing_generator = ParsingGenerator(task=args.task)
    
    # 加载输入数据
    with open(args.input, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # 处理数据
    parsing_generator.process_test_data(
        test_data=raw_data,
        output_file=args.output,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
