
import os
import json
import argparse
import time
from tqdm import tqdm
from src.parsers.parser_factory import ParserFactory
from src.verifiers.verifier_factory import VerifierFactory
from src.models.llama import LlamaModel
from src.models.openai_model import OpenaiModel
from src.utils.json_utils import JsonUtils
from src.utils.logging_utils import LoggingUtils


# 设置日志
logger = LoggingUtils.setup_logger(
    name="pipeline",
    log_file="logs/pipeline.log"
)

def load_model(model_name, model_params=None):
    """
    加载模型
    
    参数:
    - model_name: 模型名称或路径
    - model_params: 模型参数字典
    
    返回:
    - model: 模型实例
    """
    # 确保model_params是字典
    if model_params is None:
        model_params = {}
    
    # 根据模型名称选择合适的模型类
    if model_name in ['o3-mini', 'o4-mini', 'gpt-4o', 'o3', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4', 'o1']:
        # OpenAI模型
        logger.info(f"加载OpenAI模型: {model_name}")
        return OpenaiModel(model=model_name, params=model_params)
    else:
        # 默认使用Llama模型
        logger.info(f"加载Llama模型: {model_name}")
        return LlamaModel(model_path=model_name, params=model_params)
    

    

def run_pipeline(input_file, output_file, parser_type, parser_model_name, verifier_type, verifier_model_name, 
                batch_size=10, max_retries=3, parser_model_params=None, verifier_model_params=None,
                parser_kwargs=None, verifier_kwargs=None):
    """
    运行Pipeline
    
    参数:
    - input_file: 输入文件路径
    - output_file: 输出文件路径
    - parser_type: 解析器类型，可选值为"icl"或"ft"
    - parser_model_name: 解析器使用的模型名称或路径
    - verifier_type: 验证器类型，可选值为"llm"或"z3"
    - verifier_model_name: 验证器使用的模型名称或路径
    - task_type: 任务类型，可选值为"combined"、"qp"或"cp"
    - batch_size: 批处理大小
    - max_retries: 最大重试次数
    - parser_model_params: 解析器模型参数
    - verifier_model_params: 验证器模型参数
    - parser_kwargs: 传递给解析器的额外参数
    - verifier_kwargs: 传递给验证器的额外参数
    
    返回:
    - results: 处理结果
    """
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 确保参数是字典
    if parser_model_params is None:
        parser_model_params = {}
    if verifier_model_params is None:
        verifier_model_params = {}
    if parser_kwargs is None:
        parser_kwargs = {}
    if verifier_kwargs is None:
        verifier_kwargs = {}
    
    # 加载数据
    logger.info(f"加载数据: {input_file}")
    data = JsonUtils.load_json(input_file)
    
    # 加载解析器模型
    parser_model = load_model(parser_model_name, parser_model_params)
    task_type_step1 = 'combined'
    logger.info(f"创建{parser_type.upper()}解析器，任务: {task_type_step1}")
    parser = ParserFactory.create_parser(
        parser_type=parser_type,
        task_type=task_type_step1,
        model=parser_model,
        **parser_kwargs
    )
    # 设置解析结果的临时文件
    parsing_output = output_file.replace(".json", f"_parsing_step2.json")
    
    # 运行解析器
    logger.info("Step1开始解析阶段")
    parse_args = {
        "batch_size": batch_size,
        "max_retries": max_retries,
        **parser_kwargs
    }
    parsed_results = parser.parse(
        data=data,
        output_file=parsing_output,
        **parse_args
    )
    
    task_type_step2 = 'qp'
    logger.info(f"创建{parser_type.upper()}解析器，任务: {task_type_step2}")
    parser = ParserFactory.create_parser(
        parser_type=parser_type,
        task_type=task_type_step1,
        model=parser_model,
        **parser_kwargs
    )
    # 设置解析结果的临时文件
    parsing_output = output_file.replace(".json", f"_parsing_step2.json")

    logger.info("Step2开始解析阶段")
    parse_args = {
        "batch_size": batch_size,
        "max_retries": max_retries,
        **parser_kwargs
    }
    parsed_results = parser.parse(
        data=parsed_results,
        output_file=parsing_output,
        **parse_args
    )
    

    # 加载验证器模型和验证器
    verifier_model = load_model(verifier_model_name, verifier_model_params)
    verifier = VerifierFactory.create_verifier(verifier_type, verifier_model)
    
    # 运行验证器
    logger.info("开始验证阶段")
    verify_args = {
        "batch_size": batch_size,
        "max_retries": max_retries,
        **verifier_kwargs
    }
    verified_results = verifier.verify(
        data=parsed_results,
        output_file=output_file,
        **verify_args
    )
    
    logger.info(f"处理完成，结果已保存到: {output_file}")
    return verified_results

def parse_model_params(params_str):
    """
    解析模型参数字符串
    
    参数:
    - params_str: 参数字符串，格式为"key1=value1,key2=value2,..."
    
    返回:
    - params: 参数字典
    """
    if not params_str:
        return {}
    
    params = {}
    for param in params_str.split(','):
        if '=' in param:
            key, value = param.split('=', 1)
            # 尝试转换为适当的类型
            try:
                # 尝试转换为数字
                if '.' in value:
                    params[key] = float(value)
                else:
                    params[key] = int(value)
            except ValueError:
                # 如果不是数字，检查是否为布尔值
                if value.lower() == 'true':
                    params[key] = True
                elif value.lower() == 'false':
                    params[key] = False
                else:
                    # 否则保持为字符串
                    params[key] = value
    
    return params

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LLMSR Pipeline")
    parser.add_argument("--input", type=str, required=True, help="输入文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出文件路径")
    parser.add_argument("--parser", type=str, default="icl", choices=["icl", "ft"], help="解析器类型")
    parser.add_argument("--parser_model", type=str, required=True, help="解析器使用的模型名称或路径")
    parser.add_argument("--qp_parser_model_params", type=str, help="QP解析器模型参数，格式为'key1=value1,key2=value2,...'")
    parser.add_argument("--cp_parser_model_params", type=str, help="CP解析器模型参数，格式为'key1=value1,key2=value2,...'")
    parser.add_argument("--parser_model_params", type=str, help="Combined解析器模型参数，格式为'key1=value1,key2=value2,...'")
    parser.add_argument("--verifier", type=str, default="llm", choices=["llm", "z3"], help="验证器类型")
    parser.add_argument("--verifier_model", type=str, required=True, help="验证器使用的模型名称或路径")
    parser.add_argument("--verifier_model_params", type=str, help="验证器模型参数，格式为'key1=value1,key2=value2,...'")
    # parser.add_argument("--task", type=str, default="combined", choices=["combined", "qp", "cp"], help="任务类型")
    parser.add_argument("--batch_size", type=int, default=10, help="批处理大小")
    parser.add_argument("--max_retries", type=int, default=3, help="最大重试次数")
    parser.add_argument("--temperature", type=float, default=0.5, help="温度参数")
    parser.add_argument("--top_p", type=float, default=0.9, help="top_p参数")
    parser.add_argument("--max_tokens", type=int, default=1024, help="最大生成token数")
    
    args = parser.parse_args()
    
    # 解析模型参数
    parser_model_params = parse_model_params(args.parser_model_params)
    verifier_model_params = parse_model_params(args.verifier_model_params)
    
    # 添加通用模型参数
    if args.temperature is not None:
        parser_model_params.setdefault("temperature", args.temperature)
        verifier_model_params.setdefault("temperature", args.temperature)
    
    if args.top_p is not None:
        parser_model_params.setdefault("top_p", args.top_p)
        verifier_model_params.setdefault("top_p", args.top_p)
    
    if args.max_tokens is not None:
        parser_model_params.setdefault("max_tokens", args.max_tokens)
        verifier_model_params.setdefault("max_tokens", args.max_tokens)
    
    # 运行Pipeline
    run_pipeline(
        input_file=args.input,
        output_file=args.output,
        parser_type=args.parser,
        parser_model_name=args.parser_model,
        verifier_type=args.verifier,
        verifier_model_name=args.verifier_model,
        # task_type=args.task,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        parser_model_params=parser_model_params,
        verifier_model_params=verifier_model_params
    )

if __name__ == "__main__":
    # main()

    # 运行Pipeline
    results = run_pipeline(
        input_file="data/test.json",
        output_file="results/results_icl_llm.json",
        parser_type="icl",
        parser_model_name="o3-mini",
        verifier_type="llm",
        verifier_model_name="gpt-4",
        task_type="combined",
    )