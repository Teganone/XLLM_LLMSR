#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
import argparse
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
import re

def print_session(msg):
    print("-"*50)
    print(msg)
    print("-"*50)

combined_system_prompt = """You are an expert in logical parsing and reasoning analysis, specializing in analyzing problem conditions and reasoning processes. Given a question, a cot and preprocessed question_parsing and cot_parsing provided by the given question and the cot. Your task is to generate accurate question question parsing and cot parsing results based on the given question and reasoning process."""

def load_data(train_file, test_file=None):
    """加载训练和测试数据"""
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    test_data = None
    if test_file:
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    
    return train_data, test_data


def prepare_merged_data(data):
    """准备合并数据的训练样本，使用preprocessed_qp和preprocessed_cp作为输入，question_parsing和cot_parsing作为输出"""
    examples = []
    
    for item in data:
        question = item.get('question', '')
        cot = item.get('cot', '')
        preprocessed_qp = item.get('preprocessed_qp', [])
        preprocessed_cp = item.get('preprocessed_cp', [])
        question_parsing = item.get('question_parsing', [])
        cot_parsing = item.get('cot_parsing', [])
        
        # 构建系统提示和用户提示
        # system_prompt = """You are an expert in logical parsing and reasoning analysis, specializing in analyzing problem conditions and reasoning processes. Given a question, a cot and preprocessed question_parsing and cot_parsing provided by the given question and the cot. Your task is to generate accurate question question parsing and cot parsing results based on the given question and reasoning process."""
        
        user_prompt = f"""Based on the following question and chain of thought reasoning process, generate question_parsing and cot_parsing results.

Question:
{question}

Cot(Reasoning Process):
{cot}

Preprocessed Question Parsing:
{json.dumps(preprocessed_qp, ensure_ascii=False, indent=2)}

Preprocessed Cot Parsing:
{json.dumps(preprocessed_cp, ensure_ascii=False, indent=2)}

Please provide improved parsing results in the following format:
{{
  "question_parsing": [
    "condition 1",
    "condition 2",
    ...
  ],
  "cot_parsing": [
    {{
      "statement": "statement 1",
      "evidence": "evidence 1",
      "Verification": "true or false"
    }},
    {{
      "statement": "statement 2",
      "evidence": "evidence 2",
      "Verification": "true or false"
    }},
    ...
  ]
}}

Generate the improved JSON:"""
        
        assistant_response = json.dumps({
            "question_parsing": question_parsing,
            "cot_parsing": cot_parsing
        }, ensure_ascii=False)
        
        examples.append({
            "system": combined_system_prompt,
            "user": user_prompt,
            "assistant": assistant_response
        })
    
    return examples

def tokenize_function(examples, tokenizer, max_length=2048):
    """将文本转换为token，使用 Llama3 的 chat_template"""
    input_ids_list = []
    labels_list = []
    
    for system, user, assistant in zip(examples["system"], examples["user"], examples["assistant"]):
        # 构建消息列表
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant}
        ]
        
        # 应用 chat_template
        tokenized = tokenizer.apply_chat_template(
            messages, 
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        )
        
        # 创建标签，将非助手回复部分的标签设为-100（忽略）
        labels = tokenized.clone()
        
        # 找到助手回复的起始位置
        assistant_tokens = tokenizer.encode(assistant, add_special_tokens=True)
        assistant_start_idx = None
        
        # 在tokenized中查找assistant_tokens的起始位置
        for i in range(len(tokenized[0]) - len(assistant_tokens) + 1):
            if torch.all(tokenized[0][i:i+len(assistant_tokens)] == torch.tensor(assistant_tokens)):
                assistant_start_idx = i
                break
        
        if assistant_start_idx is not None:
            # 将非助手回复部分的标签设为-100
            labels[0, :assistant_start_idx] = -100
        
        input_ids_list.append(tokenized)
        labels_list.append(labels)
    
    # 填充到最大长度
    max_len = min(max([len(x[0]) for x in input_ids_list]), max_length)
    
    padded_input_ids = []
    padded_labels = []
    
    for input_ids, labels in zip(input_ids_list, labels_list):
        if len(input_ids[0]) < max_len:
            padding_length = max_len - len(input_ids[0])
            input_ids = torch.cat([input_ids, torch.zeros((1, padding_length), dtype=torch.long)], dim=1)
            labels = torch.cat([labels, torch.zeros((1, padding_length), dtype=torch.long) - 100], dim=1)
        else:
            input_ids = input_ids[:, :max_len]
            labels = labels[:, :max_len]
        
        padded_input_ids.append(input_ids[0])
        padded_labels.append(labels[0])
    
    return {
        "input_ids": torch.stack(padded_input_ids),
        "labels": torch.stack(padded_labels)
    }

def finetune(args):
    """微调模型"""
    # 加载训练数据
    train_data, test_data = load_data(args.train_file, args.test_file)
    
    # 准备训练数据
    train_examples = prepare_merged_data(train_data)
    
    # 创建数据集
    train_dataset = Dataset.from_dict({
        "system": [example["system"] for example in train_examples],
        "user": [example["user"] for example in train_examples],
        "assistant": [example["assistant"] for example in train_examples]
    })
    
    # 检测可用GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个GPU设备")
    
    # 设置设备映射策略
    device_map = args.device_map
    print(f"使用设备映射策略: {device_map}")
    
    # 加载tokenizer和模型
    try:
        # 优先使用本地模型路径
        if args.model_path:
            print(f"使用本地模型路径: {args.model_path}")
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            
            # 根据是否使用量化决定加载方式
            if args.load_8bit:
                print("使用8位量化加载模型")
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_path,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    load_in_8bit=True,
                    low_cpu_mem_usage=True
                )
            elif args.load_4bit:
                print("使用4位量化加载模型")
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_path,
                    quantization_config=bnb_config,
                    device_map=device_map,
                    low_cpu_mem_usage=True
                )
            else:
                print("使用全精度加载模型")
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_path,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    low_cpu_mem_usage=True
                )
        else:
            # 使用模型名称
            print(f"使用模型名称: {args.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            
            # 根据是否使用量化决定加载方式
            if args.load_8bit:
                print("使用8位量化加载模型")
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    load_in_8bit=True,
                    low_cpu_mem_usage=True
                )
            elif args.load_4bit:
                print("使用4位量化加载模型")
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name,
                    quantization_config=bnb_config,
                    device_map=device_map,
                    low_cpu_mem_usage=True
                )
            else:
                print("使用全精度加载模型")
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    low_cpu_mem_usage=True
                )
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("请尝试以下解决方案:")
        print("1. 检查网络连接")
        print("2. 使用--model_path参数指定本地模型路径")
        raise e
    
    tokenizer.pad_token = tokenizer.eos_token
    
    # 对数据集进行tokenize
    tokenized_train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # 准备模型进行训练
    model = prepare_model_for_kbit_training(model)
    
    # 禁用模型缓存以兼容梯度检查点
    model.config.use_cache = False
    
    model = get_peft_model(model, lora_config)
    
    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=True,
        report_to="tensorboard",
        gradient_checkpointing=True,
        optim="adamw_torch",
        bf16=False,
        max_grad_norm=1.0,   
        label_names=["labels"]  # 明确指定了 labels 是作为标签使用的字段
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    return model, tokenizer

def predict(model, tokenizer, test_data, args):
    """使用微调后的模型进行预测"""
    results = []
    
    # 分批处理测试数据，避免内存溢出
    batch_size = args.predict_batch_size
    for i in range(0, len(test_data), batch_size):
        batch_data = test_data[i:i+batch_size]
        
        for item in tqdm(batch_data, desc=f"预测批次 {i//batch_size + 1}/{(len(test_data)-1)//batch_size + 1}"):
            question = item['question']
            cot = item.get('cot', '')
            preprocessed_qp = item.get('preprocessed_qp', [])
            preprocessed_cp = item.get('preprocessed_cp', [])
            
            # 构建提示
            # system_prompt = """You are an expert in logical reasoning analysis, specializing in analyzing problem conditions and reasoning processes. Your task is to generate accurate question parsing and reasoning parsing results based on the given question and reasoning process."""
            
            user_prompt = f"""Based on the following question and reasoning process, generate question_parsing and cot_parsing results.

Question:
{question}

Cot(Reasoning Process):
{cot}

"""
            
            # 如果有预处理的解析结果，则添加到提示中
            if preprocessed_qp:
                user_prompt += f"""Preprocessed Question Parsing:
{json.dumps(preprocessed_qp, ensure_ascii=False, indent=2)}

"""
            
            if preprocessed_cp:
                user_prompt += f"""Preprocessed Cot Parsing:
{json.dumps(preprocessed_cp, ensure_ascii=False, indent=2)}

"""
            
            user_prompt += """Please provide improved parsing results in the following format:
{
  "question_parsing": [
    "condition 1",
    "condition 2",
    ...
  ],
  "cot_parsing": [
    {
      "statement": "statement 1",
      "evidence": "evidence 1",
      "Verification": "true or false"
    },
    {
      "statement": "statement 2",
      "evidence": "evidence 2",
      "Verification": "true or false"
    },
    ...
  ]
}

Generate the improved JSON:"""
            
            # 构建消息列表
            messages = [
                {"role": "system", "content": combined_system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            try:
                # 使用chat模式生成预测
                inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
                
                # 生成预测
                outputs = model.generate(
                    inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=True
                )
                
                # 清理GPU内存
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"CUDA内存不足，尝试减少输入长度或生成长度: {e}")
                    # 尝试使用更短的输入
                    shortened_prompt = truncate_prompt(user_prompt, args.max_input_length)
                    shortened_messages = [
                        {"role": "system", "content": combined_system_prompt},
                        {"role": "user", "content": shortened_prompt}
                    ]
                    inputs = tokenizer.apply_chat_template(shortened_messages, return_tensors="pt").to(model.device)
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=min(1024, args.max_new_tokens),
                        temperature=args.temperature,
                        top_p=args.top_p,
                        do_sample=True
                    )
                    torch.cuda.empty_cache()
                else:
                    raise e
            
            # 解码预测结果
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=False)
            # 提取助手回复部分
            # assistant_prefix = tokenizer.apply_chat_template([{"role": "assistant", "content": ""}], tokenize=False).strip() #多了一个<|begin_of_text|>
            assistant_prefix = '<|start_header_id|>assistant<|end_header_id|>'
            if assistant_prefix in prediction:
                prediction = prediction.split(assistant_prefix, 1)[1].strip()
            
            # 尝试提取JSON部分
            json_str = extract_json(prediction)
            
            try:
                # 解析JSON
                parsed_prediction = json.loads(json_str)
                
                # 更新结果
                result = {
                    "id": item.get('id', ''),
                    "question": question,
                    "cot": cot,
                    "answer": item.get('answer', '')
                }
                
                # 添加解析结果
                result["question_parsing"] = parsed_prediction.get("question_parsing", [])
                
                # 确保cot_parsing中的每个项目都有必要的字段
                cot_parsing = parsed_prediction.get("cot_parsing", [])
                validated_cot_parsing = []
                for entry in cot_parsing:
                    if isinstance(entry, dict) and "statement" in entry and "evidence" in entry:
                        if "Verification" not in entry:
                            entry["Verification"] = "true"  # 默认值
                        validated_cot_parsing.append(entry)
                result["cot_parsing"] = validated_cot_parsing
                
                results.append(result)
            except Exception as e:
                print_session(f"解析预测结果失败: {e}")
                print_session(f"原始输出: {prediction}")
                print_session(f"提取的JSON: {json_str}")
                # 如果解析失败，添加空结果
                results.append({
                    "id": item.get('id', ''),
                    "question": question,
                    "cot": cot,
                    "answer": item.get('answer', ''),
                    "question_parsing": [],
                    "cot_parsing": []
                })
    
    return results

def extract_json(text):
    """从文本中提取JSON部分"""
    # 尝试多种方法提取JSON
    
    # 方法1：查找第一个{和最后一个}之间的内容
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            return text[start:end+1]
    except:
        pass
    
    # 方法2：使用正则表达式查找JSON对象
    try:
        json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
        matches = re.findall(json_pattern, text)
        if matches:
            # 尝试找到最长的有效JSON
            valid_jsons = []
            for match in matches:
                try:
                    json.loads(match)
                    valid_jsons.append(match)
                except:
                    pass
            
            if valid_jsons:
                return max(valid_jsons, key=len)
    except:
        pass
    
    # 方法3：如果上述方法都失败，尝试修复常见的JSON错误
    try:
        # 移除所有非JSON部分
        lines = text.split('\n')
        json_lines = []
        in_json = False
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('{'):
                in_json = True
                json_lines.append(stripped)
            elif stripped.endswith('}'):
                json_lines.append(stripped)
                in_json = False
            elif in_json:
                json_lines.append(stripped)
        
        if json_lines:
            potential_json = ' '.join(json_lines)
            # 尝试修复常见错误
            potential_json = potential_json.replace('\'', '"')  # 单引号替换为双引号
            potential_json = re.sub(r',\s*}', '}', potential_json)  # 移除尾随逗号
            potential_json = re.sub(r',\s*]', ']', potential_json)  # 移除尾随逗号
            
            # 验证是否为有效JSON
            json.loads(potential_json)
            return potential_json
    except:
        pass
    
    # 如果所有方法都失败，返回原始文本
    return text

def truncate_prompt(prompt, max_length=2048):
    """截断提示以减少内存使用"""
    lines = prompt.split('\n')
    # 保留提示的开头和结尾部分
    header_lines = lines[:10]  # 保留前10行
    footer_lines = lines[-15:]  # 保留最后15行
    
    # 如果问题或CoT太长，截断中间部分
    if len(lines) > 25:  # 如果总行数超过25行
        middle_start = 10
        middle_end = len(lines) - 15
        middle_length = middle_end - middle_start
        
        if middle_length > 0:
            # 保留中间部分的一部分
            keep_ratio = min(1.0, (max_length - len(header_lines) - len(footer_lines)) / middle_length)
            keep_lines = int(middle_length * keep_ratio)
            
            if keep_lines > 0:
                step = middle_length / keep_lines
                indices = [middle_start + int(i * step) for i in range(keep_lines)]
                middle_lines = [lines[i] for i in indices]
            else:
                middle_lines = []
            
            return '\n'.join(header_lines + middle_lines + footer_lines)
    
    return prompt

def main():
    parser = argparse.ArgumentParser(description="使用合并数据微调Llama-3模型")
    parser.add_argument("--train_file", type=str, default="LLMSR_Datasets/Output Results/Merged_Train_result_v5_0.5_llama-3-8B-Instruct-icl.json", 
                        help="训练数据文件路径")
    parser.add_argument("--test_file", type=str, default="LLMSR_Datasets/Output Results/Merged_Test_A_result_v5_llama-3-8B-Instruct-icl.json", 
                        help="测试数据文件路径")
    parser.add_argument("--output_dir", type=str, default="/datacenter/chendanchun/models/finetuned/llama3-finetuned", 
                        help="输出目录")
    parser.add_argument("--model_name", type=str, default="LLM-Research/Meta-Llama-3-8B-Instruct", 
                        help="模型名称")
    parser.add_argument("--model_path", type=str, default='/datacenter/yingjiahao/models/Llama-3-8B-Instruct', 
                        help="本地模型路径，如果提供则优先使用")
    parser.add_argument("--max_length", type=int, default=2048, 
                        help="最大序列长度")
    parser.add_argument("--max_input_length", type=int, default=2048, 
                        help="最大输入长度（用于截断）")
    parser.add_argument("--max_new_tokens", type=int, default=1024, 
                        help="生成时的最大新token数")
    parser.add_argument("--num_epochs", type=int, default=6, 
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, 
                        help="训练批次大小")
    parser.add_argument("--predict_batch_size", type=int, default=2, 
                        help="预测批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, 
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, 
                        help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, 
                        help="预热比例")
    parser.add_argument("--logging_steps", type=int, default=6, 
                        help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=6, 
                        help="保存步数")
    parser.add_argument("--save_total_limit", type=int, default=6, 
                        help="保存的检查点总数")
    parser.add_argument("--do_train", action="store_true", 
                        help="是否进行训练")
    parser.add_argument("--do_predict", action="store_true", 
                        help="是否进行预测")
    parser.add_argument("--prediction_output", type=str, default="LLMSR_Datasets/Output Results/Finetuned_Output_result.json", 
                        help="预测结果输出文件")
    parser.add_argument("--load_8bit", action="store_true", 
                        help="是否使用8位量化加载模型")
    parser.add_argument("--load_4bit", action="store_true", 
                        help="是否使用4位量化加载模型")
    parser.add_argument("--device_map", type=str, default="auto", 
                        choices=["auto", "balanced", "sequential"], 
                        help="设备映射策略")
    parser.add_argument("--temperature", type=float, default=0.5, 
                        help="生成时的温度参数")
    parser.add_argument("--top_p", type=float, default=0.9, 
                        help="生成时的top_p参数")
    parser.add_argument("--lora_r", type=int, default=32, 
                        help="LoRA的秩")
    parser.add_argument("--lora_alpha", type=int, default=64, 
                        help="LoRA的alpha参数")
    parser.add_argument("--lora_dropout", type=float, default=0, 
                        help="LoRA的dropout率")
    parser.add_argument("--task_type", type=str, default="combined",help="qp,cp,combined")
    
    args = parser.parse_args()
    args.prediction_output = f'LLMSR_Datasets/Output Results/Finetuned_{args.test_file.split("/")[-1]}'
    args.output_dir = f"{args.output_dir}_{args.task_type}"
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.do_train:
        # 微调模型
        model, tokenizer = finetune(args)
    
    if args.do_predict:
        if not args.do_train:
            # 加载模型和tokenizer
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.output_dir, "final_model"))
            
            # 检测可用GPU数量
            num_gpus = torch.cuda.device_count()
            print(f"预测阶段检测到 {num_gpus} 个GPU设备")
            
            # 设置设备映射策略
            device_map = args.device_map
            print(f"预测阶段使用设备映射策略: {device_map}")
            
            # 根据是否使用量化决定加载方式
            if args.load_8bit:
                print("预测阶段使用8位量化加载模型")
                model = AutoModelForCausalLM.from_pretrained(
                    os.path.join(args.output_dir, "final_model"),
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    load_in_8bit=True,
                    low_cpu_mem_usage=True
                )
            elif args.load_4bit:
                print("预测阶段使用4位量化加载模型")
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
                model = AutoModelForCausalLM.from_pretrained(
                    os.path.join(args.output_dir, "final_model"),
                    quantization_config=bnb_config,
                    device_map=device_map,
                    low_cpu_mem_usage=True
                )
            else:
                print("预测阶段使用全精度加载模型")
                model = AutoModelForCausalLM.from_pretrained(
                    os.path.join(args.output_dir, "final_model"),
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    low_cpu_mem_usage=True
                )
        
        # 加载测试数据
        _, test_data = load_data(args.train_file, args.test_file)
        
        # 进行预测
        results = predict(model, tokenizer, test_data, args)
        
        # 保存预测结果
        with open(args.prediction_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"预测结果已保存到 {args.prediction_output}")

if __name__ == "__main__":
    main()
