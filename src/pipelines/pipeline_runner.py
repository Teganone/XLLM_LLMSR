import os
import json
from typing import Dict, List, Any, Optional
from src.pipelines.pipeline_builder import Pipeline
from src.utils.json_utils import JsonUtils
from src.utils.logging_utils import LoggingUtils

# 设置日志
logger = LoggingUtils.setup_logger(
    name="pipeline_runner",
    log_file="logs/pipeline_runner.log"
)

class PipelineRunner:
    """
    Pipeline 运行器，用于运行 Pipeline
    """
    
    @staticmethod
    def run(pipeline: Pipeline, input_file: str, output_file: str, **kwargs):
        """
        运行 Pipeline
        
        参数:
        - pipeline: 要运行的 Pipeline
        - input_file: 输入文件路径
        - output_file: 输出文件路径
        - **kwargs: 其他参数，可以包含：
            - batch_size: 批处理大小
            - max_retries: 最大重试次数
            - output_dir: 输出目录
        
        返回:
        - results: 处理结果
        """
        # 创建输出目录
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置输出目录参数
        kwargs.setdefault("output_dir", output_dir)
        
        # 加载数据
        logger.info(f"加载数据: {input_file}")
        data = JsonUtils.load_json(input_file)
        
        # 运行 Pipeline
        logger.info(f"运行 Pipeline: {pipeline.name}")
        results = pipeline.process(data, **kwargs)
        
        # 保存结果
        logger.info(f"保存结果: {output_file}")
        JsonUtils.save_to_file(results, output_file)
        
        return results

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
