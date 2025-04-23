#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import subprocess
import os
import sys
import logging
from datetime import datetime

os.makedirs("logs", exist_ok=True)
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)


def run_command(command):
    """执行命令并记录输出"""
    logging.info(f"执行命令: {command}")
    try:
        process = subprocess.run(
            command,
            shell=True,
            check=False
        )
        
        return_code = process.returncode
        
        if return_code != 0:
            logging.error(f"命令执行失败，返回码: {return_code}")
            return False
            
        logging.info(f"命令执行成功，返回码: {return_code}")
        return True
    except Exception as e:
        logging.error(f"执行命令时发生错误: {str(e)}")
        return False

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="LLMSR处理流水线")
    parser.add_argument("--input", type=str, required=True, help="输入文件路径")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Llama-3模型路径")
    parser.add_argument("--verify_model", type=str, default="o3", 
                        help="verify模型路径")
    parser.add_argument("--output_dir", type=str, default="results/", 
                        help="输出目录")
    args = parser.parse_args()
    print(args)
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 确保日志目录存在
    os.makedirs("logs", exist_ok=True)
    
    # 定义输出文件路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    step1_output = os.path.join(args.output_dir, f"step1_combined_{timestamp}.json")
    step2_output = os.path.join(args.output_dir, f"step2_qp_{timestamp}.json")
    step3_output = os.path.join(args.output_dir, f"step3_llama_infer_{timestamp}.json")
    final_output = os.path.join(args.output_dir, f"results_{timestamp}.json")
    
    logging.info(f"开始处理流水线，输入文件: {args.input}")
    logging.info(f"步骤1输出: {step1_output}")
    logging.info(f"步骤2输出: {step2_output}")
    logging.info(f"步骤3输出: {step3_output}")
    logging.info(f"最终输出: {final_output}")
    
    # 步骤1: 运行llama_generate.py (task_type=combined, temperature=0.5)
    logging.info("开始步骤1: 运行llama_generate.py (task_type=combined, temperature=0.5)")
    cmd1 = f"python llama_parser.py --task_type combined --temperature 0.5 --version v5 --input_file {args.input} --output_file {step1_output} --model_path {args.model_path}"
    if not run_command(cmd1):
        logging.error("步骤1失败，终止流水线")
        return
    
    # 步骤2: 运行llama_generate.py (task_type=qp, temperature=0.2)
    logging.info("开始步骤2: 运行llama_generate.py (task_type=qp, temperature=0.2)")
    cmd2 = f"python llama_parser.py --task_type qp --temperature 0.2 --version v5 --input_file {step1_output} --output_file {step2_output} --model_path {args.model_path}"
    if not run_command(cmd2):
        logging.error("步骤2失败，终止流水线")
        return
    
    # 步骤3: 运行inference_llama.py
    logging.info("开始步骤3: 运行inference_llama.py")
    cmd3 = f"python llama_verifier.py --input {step2_output} --output {step3_output} --temperature 0.1 --model_path {args.model_path} --batch_mode"
    if not run_command(cmd3):
        logging.error("步骤3失败，终止流水线")
        return

    # step3_output = os.path.join(args.output_dir, "step3_llama_infer_20250417_001800.json")

    args.verify_model = 'o4-mini'
    # 步骤4: 运行inference_gpt.py
    logging.info("开始步骤4: 运行inference_gpt.py")
    cmd4 = f"python gpt_verifier.py --input {step3_output} --output {final_output} --model {args.verify_model}"
    if not run_command(cmd4):
        logging.error("步骤4失败，终止流水线")
        return
    logging.info(f"流水线执行完成！最终结果保存在: {final_output}")

if __name__ == "__main__":
    main()
