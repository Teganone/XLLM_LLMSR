from src.pipelines.pipeline_builder import PipelineBuilder
from src.pipelines.pipeline_runner import PipelineRunner, parse_model_params
from src.utils.logging_utils import LoggingUtils
import argparse
from datetime import datetime



def best_pipeline():
    parser = argparse.ArgumentParser(description="LLMSR Pipeline Runner")
    parser.add_argument("--input", type=str, required=True, help="输入文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出文件路径")
    parser.add_argument("--model", type=str, required=True, help="Llama文件路径")
    args = parser.parse_args()


    # 创建Pipeline构建器
    builder = PipelineBuilder("LLMSR_Pipeline")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # model_path = '/datacenter/models/LLM-Research/Llama-3-8B-Instruct'
    model_path = args.model
    
    # 添加组合解析节点 (--combined --combined_parser icl --combined_model gpt-4)
    builder.add_parser(
        name="combined_parser",
        parser_type="icl",  # 对应 --combined_parser icl
        model_name=model_path, # 对应 --combined_model gpt-4
        task_type="combined", # 对应 --combined
        model_params={
            "temperature": 0.6,
            "top_p": 0.9,
        },
        output_file=f"results/step1_results_combined_{timestamp}.json"
    )
    
    # 添加问题解析节点 (--qp --qp_parser icl --qp_model gpt-4)
    builder.add_parser(
        name="qp_parser",
        parser_type="icl",  # 对应 --qp_parser icl
        model_name=model_path, # 对应 --qp_model gpt-4
        task_type="qp",     # 对应 --qp
        model_params={
            "temperature": 0.2,
            "top_p": 0.8,
            "do_sample": False
        },
        output_file=f"results/step2_results_qp_{timestamp}.json"
    )
    
    # 添加验证节点 (--verify --verifier llm --verifier_model gpt-4)
    builder.add_verifier(
        name="verifier",
        verifier_type="llm",     # 对应 --verifier llm
        model_name=model_path,      # 对应 --verifier_model gpt-4
        model_params={
            "temperature": 0.2,
            # "top_p": 0.9,
            # "reasoning_effort": 'high',
        },
        output_file=f"results/step3_results_llama_verified.json"
    )
    
    # 添加验证节点 (--verify --verifier llm --verifier_model gpt-4)
    builder.add_verifier(
        name="verifier",
        verifier_type="z3",     # 对应 --verifier llm
        model_name="o3-mini",      # 对应 --verifier_model gpt-4
        model_params={
            # "temperature": 0.5,
            # "top_p": 0.9,
            "reasoning_effort": 'high',
        },
        # output_file=f"results/results_verified.json"
    )
    
    # 构建Pipeline
    pipeline = builder.build()
    
    # 运行Pipeline (--input data/test.json --output results/results_full.json)
    results = PipelineRunner.run(
        pipeline=pipeline,
        input_file=args.input,        # 对应 --input data/test.json
        # output_file=f"results/final_results_{timestamp}.json", # 对应 --output results/results_full.json
        output_file=args.output, # 对应 --output results/results_full.json
        batch_size=10,
        max_retries=3
    )
    
    return results

if __name__ == "__main__":
    best_pipeline()
