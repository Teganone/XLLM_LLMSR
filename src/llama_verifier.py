import os
import sys
sys.path.append('./')
import json
import re
import time
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 默认模型路径
# DEFAULT_MODEL_PATH = '/datacenter/yingjiahao/models/Llama-3-8B-Instruct'

def get_llama_validation_response(prompt, text, model_path, pipe=None, tokenizer=None, temperature=0.1, max_retries=3, retry_delay=2):
    """
    使用Llama-3模型获取验证响应
    
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
                max_new_tokens=128,  # 对于验证任务，较短的输出就足够了
                temperature=temperature,
                do_sample=False,  # 对于验证任务，我们希望确定性输出
                top_p=0.9,
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
                return ""

def process_validation_inference(test_data, output_file, model_path, batch_size=10, temperature=0.1):
    """
    使用Llama-3模型处理验证推理任务
    
    Args:
        test_data: 测试数据
        output_file: 输出文件路径
        model_path: 模型路径
        batch_size: 批处理大小
        temperature: 温度参数
    """
    results = []
    
    # 加载模型和tokenizer（只加载一次）
    print(f"正在加载模型 {model_path}...")
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
    print("模型加载完成！")
    
    prompt = 'Whether the "statement" can be deduced from the "evidence" logically, answer with only with True or False, do not output other contents.'
    
    for item in tqdm(test_data, desc="处理测试数据"):
        id = item.get('id', '')
        text_pre = 'question: ' + item['question'] + "\n"
        
        for sev in item.get('cot_parsing', []):
            try:
                text = text_pre + 'statement: ' + sev['statement'] + ' ' + 'evidence: ' + sev['evidence']
                response = get_llama_validation_response(prompt, text, model_path, pipe, tokenizer, temperature)
                
                if not isinstance(response, str):
                    print("response not str:", response)
                    if 'Verification' not in sev or sev['Verification'] not in ['true', 'True', 'False', 'false']:
                        sev['Verification'] = "false"
                    continue
                
                true_match = re.search(r'\btrue\b', response.lower())
                false_match = re.search(r'\bfalse\b', response.lower())
                
                if true_match and not false_match:
                    sev['Verification'] = "true"
                elif false_match and not true_match:
                    sev['Verification'] = "false"
                elif true_match and false_match:
                    # 如果同时包含true和false，可以基于它们的位置或上下文做决定
                    # 这里简单地取第一个出现的
                    if response.lower().find('true') < response.lower().find('false'):
                        sev['Verification'] = "true"
                    else:
                        sev['Verification'] = "false"
                else:
                    print("answer not true or false:", response)
                    if 'Verification' not in sev or sev['Verification'] not in ['true', 'True', 'False', 'false']:
                        sev['Verification'] = "false"
            except Exception as e:
                print(f"处理验证失败: {str(e)}")
                if 'Verification' not in sev:
                    sev['Verification'] = "false"
        
        # 构建结果
        result = {
            "question": item['question'],
            "question_parsing": item.get("question_parsing", []),
            "answer": item.get("answer", ""),
            "id": id,
            "cot": item.get("cot", ""),
            "cot_parsing": item.get("cot_parsing", []),
        }
        
        # 如果原始数据中有sel_idx字段，也添加到结果中
        if 'sel_idx' in item:
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

def process_validation_inference_batch(test_data, output_file, model_path, batch_size=10, temperature=0.1):
    """
    使用Llama-3模型批量处理验证推理任务（优化版本）
    
    Args:
        test_data: 测试数据
        output_file: 输出文件路径
        model_path: 模型路径
        batch_size: 批处理大小
        temperature: 温度参数
    """
    results = []
    
    # 加载模型和tokenizer（只加载一次）
    print(f"正在加载模型 {model_path}...")
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
    print("模型加载完成！")
    
    prompt = 'Whether the "statement" can be deduced from the "evidence" logically, answer with only with True or False, do not output other contents.'
    
    # 处理每个测试样本
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i+batch_size]
        batch_results = []
        
        for item in tqdm(batch, desc=f"处理批次 {i//batch_size + 1}/{(len(test_data)-1)//batch_size + 1}"):
            id = item.get('id', '')
            text_pre = 'question: ' + item['question'] + "\n"
            
            for sev in item.get('cot_parsing', []):
                try:
                    text = text_pre + 'statement: ' + sev['statement'] + ' ' + 'evidence: ' + sev['evidence']
                    response = get_llama_validation_response(prompt, text, model_path, pipe, tokenizer, temperature)
                    
                    if not isinstance(response, str):
                        print("response not str:", response)
                        if 'Verification' not in sev or sev['Verification'] not in ['true', 'True', 'False', 'false']:
                            sev['Verification'] = "false"
                        continue
                    
                    true_match = re.search(r'\btrue\b', response.lower())
                    false_match = re.search(r'\bfalse\b', response.lower())
                    
                    if true_match and not false_match:
                        sev['Verification'] = "true"
                    elif false_match and not true_match:
                        sev['Verification'] = "false"
                    elif true_match and false_match:
                        # 如果同时包含true和false，可以基于它们的位置或上下文做决定
                        # 这里简单地取第一个出现的
                        if response.lower().find('true') < response.lower().find('false'):
                            sev['Verification'] = "true"
                        else:
                            sev['Verification'] = "false"
                    else:
                        print("answer not true or false:", response)
                        if 'Verification' not in sev or sev['Verification'] not in ['true', 'True', 'False', 'false']:
                            sev['Verification'] = "false"
                except Exception as e:
                    print(f"处理验证失败: {str(e)}")
                    if 'Verification' not in sev:
                        sev['Verification'] = "false"
            
            # 构建结果
            result = {
                "question": item['question'],
                "question_parsing": item.get("question_parsing", []),
                "answer": item.get("answer", ""),
                "id": id,
                "cot": item.get("cot", ""),
                "cot_parsing": item.get("cot_parsing", []),
            }
            
            # 如果原始数据中有sel_idx字段，也添加到结果中
            if 'sel_idx' in item:
                result['sel_idx'] = item['sel_idx']
            
            batch_results.append(result)
        
        # 更新结果列表
        results.extend(batch_results)
        
        # 每处理一个批次，保存一次结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"已保存{len(results)}个结果到{output_file}")
        
        # 清理GPU内存
        torch.cuda.empty_cache()
    
    print(f"处理完成，共{len(results)}个结果已保存到{output_file}")

def main(args):
    """
    主函数，处理命令行参数并执行验证任务
    
    Args:
        args: 命令行参数
    """
    # 加载输入数据
    with open(args.input, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # 根据批处理模式选择处理函数
    if args.batch_mode:
        process_validation_inference_batch(
            test_data=raw_data,
            output_file=args.output,
            model_path=args.model_path,
            batch_size=args.batch_size,
            temperature=args.temperature
        )
    else:
        process_validation_inference(
            test_data=raw_data,
            output_file=args.output,
            model_path=args.model_path,
            batch_size=args.batch_size,
            temperature=args.temperature
        )

def parse_args():
    """
    解析命令行参数
    
    Returns:
        解析后的参数
    """
    import argparse
    parser = argparse.ArgumentParser(description="使用Llama-3模型进行验证推理")
    parser.add_argument("--input", type=str, 
                        required=True, 
                        help="输入JSON文件")
    parser.add_argument("--output", type=str, 
                        required=True, 
                        help="输出JSON文件")
    parser.add_argument("--model_path", type=str, 
                        required=True, 
                        help="Llama-3模型路径")
    parser.add_argument("--batch_size", type=int, 
                        default=5, 
                        help="批处理大小")
    parser.add_argument("--temperature", type=float, 
                        default=0.1, 
                        help="生成温度参数")
    parser.add_argument("--batch_mode", action="store_true", 
                        help="是否使用批处理模式（优化内存使用）")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
