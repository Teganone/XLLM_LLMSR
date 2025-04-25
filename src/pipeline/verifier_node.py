import os
from typing import Dict, List, Any, Optional
from src.pipeline.node import Node
from src.verifiers.verifier_factory import VerifierFactory
from src.models.llama import LlamaModel
from src.models.openai_model import OpenaiModel
from src.utils.json_utils import JsonUtils
from src.utils.logging_utils import LoggingUtils

# 设置日志
logger = LoggingUtils.setup_logger(
    name="verifier_node",
    log_file="logs/verifier_node.log"
)

class VerifierNode(Node):
    """
    验证器节点
    """
    
    def __init__(self, name: str, verifier_type: str, model_name: str, 
                 model_params: Optional[Dict[str, Any]] = None, verifier_kwargs: Optional[Dict[str, Any]] = None,
                 output_file: Optional[str] = None):
        """
        初始化验证器节点
        
        参数:
        - name: 节点名称
        - verifier_type: 验证器类型，可选值为"llm"或"z3"
        - model_name: 模型名称或路径
        - model_params: 模型参数
        - verifier_kwargs: 传递给验证器的额外参数
        - output_file: 输出文件路径（可选）
        """
        super().__init__(name)
        self.verifier_type = verifier_type
        self.model_name = model_name
        self.model_params = model_params or {}
        self.verifier_kwargs = verifier_kwargs or {}
        self.output_file = output_file
        self.verifier = None
    
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
        
        # 如果验证器未初始化，则初始化
        if self.verifier is None:
            # 加载模型
            model = self._load_model()
            
            # 创建验证器
            logger.info(f"创建{self.verifier_type.upper()}验证器")
            self.verifier = VerifierFactory.create_verifier(
                verifier_type=self.verifier_type,
                model=model,
                **self.verifier_kwargs
            )
        
        # 设置输出文件
        output_file = self.output_file
        if output_file is None and "output_dir" in kwargs:
            output_file = os.path.join(kwargs["output_dir"], f"{self.name}.json")
        
        # 运行验证器
        logger.info(f"运行{self.name}验证器")
        verify_args = {
            "batch_size": batch_size,
            "max_retries": max_retries
        }
        processed_data = self.verifier.verify(
            data=data,
            output_file=output_file,
            **verify_args
        )
        
        return processed_data
