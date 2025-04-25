
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
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
import re
from typing import List, Dict, Any, Optional, Union
from src.utils.json_utils import JsonUtils
from src.utils.logging_utils import LoggingUtils

# 设置日志
logger = LoggingUtils.setup_logger(
    name="finetune",
    log_file="logs/finetune.log"
)

class FinetuneParser:
    """用于微调和预测的解析器类"""
    
    def __init__(self, model_path=None, task_type="combined"):
        """
        初始化微调解析器
        
        参数:
        - model_path: 模型路径（优先使用）
        - model_name: 模型名称（如果model_path为None则使用）
        - task_type: 任务类型，可选值为"combined"、"qp"、"cp"
        """
        self.model_path = model_path
        # self.model_name = model_name
        self.task_type = task_type
        self.model = None
        self.tokenizer = None
        self.load_prompt_templates()

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
    
    
    def load_model(self, device_map="auto", load_8bit=False, load_4bit=True):
        """
        加载模型和分词器
        
        参数:
        - device_map: 设备映射策略
        - load_8bit: 是否使用8位量化
        - load_4bit: 是否使用4位量化
        
        返回:
        - model: 加载的模型
        - tokenizer: 加载的分词器
        """
        try:
            # 确定模型路径或名称
            # model_source = self.model_path if self.model_path else self.model_name
            model_source = self.model_path
            if not model_source:
                raise ValueError("必须提供model_path或model_name")
            
            logger.info(f"加载模型: {model_source}")
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(model_source)
            
            # 确保tokenizer有正确的pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 根据是否使用量化决定加载方式
            if load_8bit:
                logger.info("使用8位量化加载模型")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_source,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    load_in_8bit=True,
                    low_cpu_mem_usage=True
                )
            elif load_4bit:
                logger.info("使用4位量化加载模型")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_source,
                    quantization_config=bnb_config,
                    device_map=device_map,
                    low_cpu_mem_usage=True
                )
            else:
                logger.info("使用全精度加载模型")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_source,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    low_cpu_mem_usage=True
                )
            
            logger.info("模型加载完成")
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def _prepare_prompts(self, test_data):
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
        assistant_response = JsonUtils.format_json({
            "question_parsing": test_data.get('question_parsing', '[]'),
            "cot_parsing": test_data.get('cot_parsing', [])
        })
            
        return user_prompt, assistant_response
        
    def _prepare_data(self, data):
        """
        准备训练或预测数据
        
        参数:
        - data: 原始数据
        - include_preprocessed: 是否包含预处理的解析结果
        
        返回:
        - examples: 准备好的数据样本
        """
        examples = []
        
        for item in data:
            user_prompt, assistant_response = self._prepare_prompts(item)
            examples.append({
                "system": self.system_prompt,
                "user": user_prompt,
                "assistant": assistant_response
            })
        
        return examples
    

    
    def tokenize_data(self, examples, max_length=2048):
        """
        将文本转换为token
        
        参数:
        - examples: 数据样本
        - max_length: 最大序列长度
        
        返回:
        - tokenized_data: 分词后的数据
        """
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
            tokenized = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt"
            )
            
            # 创建标签，将非助手回复部分的标签设为-100（忽略）
            labels = tokenized.clone()
            
            # 找到助手回复的起始位置
            assistant_tokens = self.tokenizer.encode(assistant, add_special_tokens=True)
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
    
    def finetune(self, train_data, output_dir, training_args=None, lora_config=None):
        """
        微调模型
        
        参数:
        - train_data: 训练数据
        - output_dir: 输出目录
        - training_args: 训练参数
        - lora_config: LoRA配置
        
        返回:
        - model: 微调后的模型
        - tokenizer: 分词器
        """
        # 准备训练数据
        train_examples = self.prepare_data(train_data)
        
        # 创建数据集
        train_dataset = Dataset.from_dict({
            "system": [example["system"] for example in train_examples],
            "user": [example["user"] for example in train_examples],
            "assistant": [example["assistant"] for example in train_examples]
        })
        
        # 对数据集进行tokenize
        tokenized_train_dataset = train_dataset.map(
            lambda examples: self.tokenize_data(examples),
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        # 默认LoRA配置
        if lora_config is None:
            lora_config = LoraConfig(
                r=32,
                lora_alpha=64,
                lora_dropout=0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
        
        # 准备模型进行训练
        model = prepare_model_for_kbit_training(self.model)
        
        # 禁用模型缓存以兼容梯度检查点
        model.config.use_cache = False
        
        model = get_peft_model(model, lora_config)
        
        # 默认训练参数
        if training_args is None:
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=6,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=2,
                learning_rate=2e-5,
                weight_decay=0.01,
                warmup_ratio=0.1,
                logging_steps=6,
                save_steps=6,
                save_total_limit=6,
                fp16=True,
                report_to="tensorboard",
                gradient_checkpointing=True,
                optim="adamw_torch",
                bf16=False,
                max_grad_norm=1.0,
                label_names=["labels"]
            )
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        )
        
        # 开始训练
        logger.info("开始训练...")
        trainer.train()
        
        # 保存模型
        model_output_dir = os.path.join(output_dir, "final_model")
        model.save_pretrained(model_output_dir)
        self.tokenizer.save_pretrained(model_output_dir)
        logger.info(f"模型已保存到 {model_output_dir}")
        
        return model, self.tokenizer
    
    def predict(self, test_data, batch_size=10, max_new_tokens=1024, temperature=0.5, top_p=0.9, max_input_length=2048):
        """
        使用微调后的模型进行预测
        
        参数:
        - test_data: 测试数据
        - batch_size: 批处理大小
        - max_new_tokens: 生成的最大新token数
        - temperature: 温度参数
        - top_p: top-p采样参数
        - max_input_length: 最大输入长度（用于截断）
        
        返回:
        - results: 预测结果
        """
        results = []
        
        # 分批处理测试数据，避免内存溢出
        for i in range(0, len(test_data), batch_size):
            batch_data = test_data[i:i+batch_size]
            
            for item in tqdm(batch_data, desc=f"预测批次 {i//batch_size + 1}/{(len(test_data)-1)//batch_size + 1}"):
                # 准备单个样本的数据
                examples = self.prepare_data([item], include_preprocessed=True)
                if not examples:
                    continue
                
                example = examples[0]
                
                # 构建消息列表
                messages = [
                    {"role": "system", "content": example["system"]},
                    {"role": "user", "content": example["user"]}
                ]
                
                try:
                    # 使用chat模式生成预测
                    inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.model.device)
                    
                    # 生成预测
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True
                    )
                    
                    # 清理GPU内存
                    torch.cuda.empty_cache()
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        logger.warning(f"CUDA内存不足，尝试减少输入长度或生成长度: {e}")
                        # 尝试使用更短的输入
                        shortened_prompt = self._truncate_prompt(example["user"], max_input_length)
                        shortened_messages = [
                            {"role": "system", "content": example["system"]},
                            {"role": "user", "content": shortened_prompt}
                        ]
                        inputs = self.tokenizer.apply_chat_template(shortened_messages, return_tensors="pt").to(self.model.device)
                        outputs = self.model.generate(
                            inputs,
                            max_new_tokens=min(1024, max_new_tokens),
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=True
                        )
                        torch.cuda.empty_cache()
                    else:
                        logger.error(f"生成预测时出错: {e}")
                        continue
                
                # 解码预测结果
                prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
                
                # 提取助手回复部分
                assistant_prefix = '<|start_header_id|>assistant<|end_header_id|>'
                if assistant_prefix in prediction:
                    prediction = prediction.split(assistant_prefix, 1)[1].strip()
                else:
                    # 尝试通过比较输入和输出长度来提取回复
                    prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
                    if len(prediction) > len(prompt_text):
                        prediction = prediction[len(prompt_text):].strip()
                
                # 尝试提取JSON部分
                json_str = self._extract_json(prediction)
                
                try:
                    # 解析JSON
                    parsed_prediction = json.loads(json_str)
                    
                    # 更新结果
                    result = {
                        "id": item.get('id', ''),
                        "question": item.get('question', ''),
                        "cot": item.get('cot', ''),
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
                    logger.error(f"解析预测结果失败: {e}")
                    logger.error(f"原始输出: {prediction}")
                    logger.error(f"提取的JSON: {json_str}")
                    # 如果解析失败，添加空结果
                    results.append({
                        "id": item.get('id', ''),
                        "question": item.get('question', ''),
                        "cot": item.get('cot', ''),
                        "answer": item.get('answer', ''),
                        "question_parsing": [],
                        "cot_parsing": []
                    })
        
        return results
    
    def _extract_json(self, text):
        """
        从文本中提取JSON部分
        
        参数:
        - text: 文本
        
        返回:
        - json_str: JSON字符串
        """
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
    
    def _truncate_prompt(self, prompt, max_length=2048):
        """
        截断提示以减少内存使用
        
        参数:
        - prompt: 提示
        - max_length: 最大长度
        
        返回:
        - truncated_prompt: 截断后的提示
        """
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
    
    def save_results(self, results, output_file):
        """
        保存结果到文件
        
        参数:
        - results: 结果
        - output_file: 输出文件路径
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存到 {output_file}")

def main():
    parser = argparse.ArgumentParser(description="微调和预测解析器")
    parser.add_argument("--train_file", type=str, help="训练数据文件路径")
    parser.add_argument("--test_file", type=str, help="测试数据文件路径")
    parser.add_argument("--output_dir", type=str, default="models/finetuned", help="输出目录")
    parser.add_argument("--model_name", type=str, help="模型名称")
    parser.add_argument("--model_path", type=str, help="本地模型路径，如果提供则优先使用")
    parser.add_argument("--task_type", type=str, default="combined", choices=["combined", "qp", "cp"], help="任务类型,目前只支持combined")
    parser.add_argument("--do_train", action="store_true", help="是否进行训练")
    parser.add_argument("--do_predict", action="store_true", help="是否进行预测")
    parser.add_argument("--prediction_output", type=str, help="预测结果输出文件")
    parser.add_argument("--load_8bit", action="store_true", help="是否使用8位量化加载模型")
    parser.add_argument("--load_4bit", action="store_true", help="是否使用4位量化加载模型")
    parser.add_argument("--device_map", type=str, default="auto", choices=["auto", "balanced", "sequential"], help="设备映射策略")
    parser.add_argument("--batch_size", type=int, default=10, help="批处理大小")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="生成的最大新token数")
    parser.add_argument("--temperature", type=float, default=0.5, help="生成时的温度参数")
    parser.add_argument("--top_p", type=float, default=0.9, help="生成时的top_p参数")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = f"{args.output_dir}_{args.task_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建微调解析器
    finetune_parser = FinetuneParser(
        model_path=args.model_path,
        model_name=args.model_name,
        task_type=args.task_type
    )
    
    # 加载模型
    finetune_parser.load_model(
        device_map=args.device_map,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit
    )
    
    if args.do_train:
        # 加载训练数据
        with open(args.train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        # 微调模型
        finetune_parser.finetune(train_data, output_dir)
    
    if args.do_predict:
        # 加载测试数据
        with open(args.test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # 进行预测
        results = finetune_parser.predict(
            test_data,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        # 保存预测结果
        prediction_output = args.prediction_output or f"results/finetuned_{args.task_type}_results.json"
        finetune_parser.save_results(results, prediction_output)

if __name__ == "__main__":
    main()
    from src.parsers.finetune import FinetuneParser
    import json

    # 加载训练数据
    with open("data/train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)

    # 创建微调解析器
    finetune_parser = FinetuneParser(
        model_path="/path/to/llama/model",
        task_type="combined"
    )

    # 加载模型
    finetune_parser.load_model(load_8bit=True)

    # 微调模型
    finetune_parser.finetune(train_data, "models/finetuned_combined")