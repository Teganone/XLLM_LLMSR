#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from src.pipelines.pipeline_builder import PipelineBuilder
from src.pipelines.pipeline_runner import PipelineRunner, parse_model_params
from src.utils.logging_utils import LoggingUtils
import subprocess
from datetime import datetime



# 设置日志
logger = LoggingUtils.setup_logger(
    name="run_pipeline",
    log_file="logs/run_pipeline.log"
)


def run_command(command):
    """执行命令并记录输出"""
    logger.info(f"执行命令: {command}")
    try:
        process = subprocess.run(
            command,
            shell=True,
            check=False
        )
        
        return_code = process.returncode
        
        if return_code != 0:
            logger.error(f"命令执行失败，返回码: {return_code}")
            return False
            
        logger.info(f"命令执行成功，返回码: {return_code}")
        return True
    except Exception as e:
        logger.error(f"执行命令时发生错误: {str(e)}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LLMSR Pipeline Runner")
    parser.add_argument("--input", type=str, required=True, help="输入文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出文件路径")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--batch_size", type=int, default=10, help="批处理大小")
    parser.add_argument("--max_retries", type=int, default=3, help="最大重试次数")
    
    # 解析器参数verif
    parser.add_argument("--combined", action="store_true", help="是否包含组合解析阶段")
    parser.add_argument("--combined_parser", type=str, default="icl", choices=["icl", "ft"], help="组合解析器类型")
    parser.add_argument("--combined_model", type=str, default="gpt-4", help="组合解析器使用的模型名称或路径")
    parser.add_argument("--combined_params", type=str, help="组合解析器模型参数，格式为'key1=value1,key2=value2,...'")
    
    parser.add_argument("--qp", action="store_true", help="是否包含问题解析阶段")
    parser.add_argument("--qp_parser", type=str, default="icl", choices=["icl", "ft"], help="问题解析器类型")
    parser.add_argument("--qp_model", type=str, default="gpt-4", help="问题解析器使用的模型名称或路径")
    parser.add_argument("--qp_params", type=str, help="问题解析器模型参数，格式为'key1=value1,key2=value2,...'")
    
    parser.add_argument("--cp", action="store_true", help="是否包含思维链解析阶段")
    parser.add_argument("--cp_parser", type=str, default="icl", choices=["icl", "ft"], help="思维链解析器类型")
    parser.add_argument("--cp_model", type=str, default="gpt-4", help="思维链解析器使用的模型名称或路径")
    parser.add_argument("--cp_params", type=str, help="思维链解析器模型参数，格式为'key1=value1,key2=value2,...'")
    
    # 验证器参数
    parser.add_argument("--verify", action="store_true", help="是否包含验证阶段")
    parser.add_argument("--verifier", type=str, default="llm", choices=["llm", "z3"], help="验证器类型")
    parser.add_argument("--verifier_model", type=str, default="gpt-4", help="验证器使用的模型名称或路径")
    parser.add_argument("--verifier_params", type=str, help="验证器模型参数，格式为'key1=value1,key2=value2,...'")
    
    # 通用模型参数
    parser.add_argument("--temperature", type=float, default=0.7, help="温度参数")
    parser.add_argument("--top_p", type=float, default=0.9, help="top_p参数")
    parser.add_argument("--max_tokens", type=int, default=1024, help="最大生成token数")
    
    args = parser.parse_args()
    logger.info(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 如果提供了配置文件，从配置文件加载参数
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = {}
    
    # 解析模型参数
    combined_params = parse_model_params(args.combined_params)
    qp_params = parse_model_params(args.qp_params)
    cp_params = parse_model_params(args.cp_params)
    verifier_params = parse_model_params(args.verifier_params)
    
    # 添加通用模型参数
    if args.temperature is not None:
        combined_params.setdefault("temperature", args.temperature)
        qp_params.setdefault("temperature", args.temperature)
        cp_params.setdefault("temperature", args.temperature)
        verifier_params.setdefault("temperature", args.temperature)
    
    if args.top_p is not None:
        combined_params.setdefault("top_p", args.top_p)
        qp_params.setdefault("top_p", args.top_p)
        cp_params.setdefault("top_p", args.top_p)
        verifier_params.setdefault("top_p", args.top_p)
    
    if args.max_tokens is not None:
        combined_params.setdefault("max_new_tokens", args.max_tokens)
        qp_params.setdefault("max_new_tokens", args.max_tokens)
        cp_params.setdefault("max_new_tokens", args.max_tokens)
        verifier_params.setdefault("max_new_tokens", args.max_tokens)
    
    # 创建 Pipeline 构建器
    builder = PipelineBuilder("LLMSR_Pipeline")
    
    # 按照 combined -> qp -> cp -> verify 的顺序添加节点
    # 这样可以确保先进行组合解析，然后是问题解析和思维链解析，最后是验证
    step = 1
    # 添加组合解析节点
    if args.combined or config.get("combined", False):
        builder.add_parser(
            name="combined_parser",
            parser_type=args.combined_parser,
            model_name=args.combined_model,
            task_type="combined",
            model_params=combined_params,
            output_file=args.output.replace(".json", f"_combined_step{step}_{timestamp}.json")
        )
        step = step + 1
    
    # 添加问题解析节点
    if args.qp or config.get("qp", False):
        builder.add_parser(
            name="qp_parser",
            parser_type=args.qp_parser,
            model_name=args.qp_model,
            task_type="qp",
            model_params=qp_params,
            output_file=args.output.replace(".json", f"_qp_step{step}_{timestamp}.json")
        )
        step = step + 1
    
    # 添加思维链解析节点
    if args.cp or config.get("cp", False):
        builder.add_parser(
            name="cp_parser",
            parser_type=args.cp_parser,
            model_name=args.cp_model,
            task_type="cp",
            model_params=cp_params,
            output_file=args.output.replace(".json", f"_cp_step{step}_{timestamp}.json")
        )
        step = step + 1
    
    # 添加验证节点
    if args.verify or config.get("verify", False):
        builder.add_verifier(
            name="verifier",
            verifier_type=args.verifier,
            model_name=args.verifier_model,
            model_params=verifier_params,
            output_file=args.output.replace('.json',f'_final_{timestamp}.json')
        )
    
    # 构建 Pipeline
    pipeline = builder.build()
    
    # 运行 Pipeline
    PipelineRunner.run(
        pipeline=pipeline,
        input_file=args.input,
        output_file=args.output,
        batch_size=args.batch_size,
        max_retries=args.max_retries
    )


if __name__ == "__main__":
    main()
    # python -m src.llmsr_pipeline --input data/train.json --output results/train_results.json --combined --combined_parser icl --combined_model '/datacenter/models/LLM-Research/Llama-3-8B-Instruct' --combined_params 'temperature=0.5' --qp --qp_parser icl --qp_model '/datacenter/models/LLM-Research/Llama-3-8B-Instruct' --qp_params 'temperature=0.2' --verify --verifier llm --verifier_model o3-mini --verifier_params 'reasoning_effort=high' 