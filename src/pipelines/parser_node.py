import os
from typing import Dict, List, Any, Optional
from src.pipelines.node import Node
from src.parsers.parser_factory import ParserFactory
from src.models.llama import LlamaModel
from src.models.openai_model import OpenaiModel
from src.utils.json_utils import JsonUtils
from src.utils.logging_utils import LoggingUtils

# 设置日志
logger = LoggingUtils.setup_logger(
    name="parser_node",
    log_file="logs/parser_node.log"
)

class ParserNode(Node):
    """
    解析器节点
    """
    
    def __init__(self, name: str, parser_type: str, model_name: str, task_type: str, 
                 model_params: Optional[Dict[str, Any]] = None, parser_kwargs: Optional[Dict[str, Any]] = None,
                 output_file: Optional[str] = None):
        """
        初始化解析器节点
        
        参数:
        - name: 节点名称
        - parser_type: 解析器类型，可选值为"icl"或"ft"
        - model_name: 模型名称或路径
        - task_type: 任务类型，可选值为"combined"、"qp"或"cp"
        - model_params: 模型参数
        - parser_kwargs: 传递给解析器的额外参数
        - output_file: 输出文件路径（可选）
        """
        super().__init__(name)
        self.parser_type = parser_type
        self.model_name = model_name
        self.task_type = task_type
        self.model_params = model_params or {}
        self.parser_kwargs = parser_kwargs or {}
        self.output_file = output_file
        self.parser = None
    
    def _load_model(self):
        """
        加载模型
        
        返回:
        - model: 模型实例
        """
        # 根据模型名称选择合适的模型类
        if self.model_name in ['o3-mini', 'o4-mini', 'gpt-4o', 'o3', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4', 'o1']:
            # OpenAI模型
            logger.info(f"加载OpenAI模型: {self.model_name}")
            return OpenaiModel(model=self.model_name, params=self.model_params)
        else:
            # 默认使用Llama模型
            logger.info(f"加载Llama模型: {self.model_name}")
            return LlamaModel(model_path=self.model_name, params=self.model_params)
    
    def process(self, data: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        处理数据
        
        参数:
        - data: 输入数据
        - **kwargs: 其他参数，可以包含：
            - batch_size: 批处理大小
            - max_retries: 最大重试次数
        
        返回:
        - processed_data: 处理后的数据
        """
        # 获取参数
        batch_size = kwargs.get("batch_size", 10)
        max_retries = kwargs.get("max_retries", 3)
        
        # 如果解析器未初始化，则初始化
        if self.parser is None:
            # 加载模型
            model = self._load_model()
            
            # 创建解析器
            logger.info(f"创建{self.parser_type.upper()}解析器，任务: {self.task_type}")
            self.parser = ParserFactory.create_parser(
                parser_type=self.parser_type,
                model=model,
                task_type=self.task_type,
                **self.parser_kwargs
            )
        
        # 设置输出文件
        output_file = self.output_file
        if output_file is None and "output_dir" in kwargs:
            output_file = os.path.join(kwargs["output_dir"], f"{self.name}.json")
        
        # 运行解析器
        logger.info(f"运行{self.name}解析器")
        parse_args = {
            "batch_size": batch_size,
            "max_retries": max_retries
        }
        processed_data = self.parser.parse(
            data=data,
            output_file=output_file,
            **parse_args
        )
        
        return processed_data
