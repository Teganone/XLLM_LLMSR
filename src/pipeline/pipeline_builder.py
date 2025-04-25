from typing import Dict, List, Any, Optional
from src.pipeline.node import Node
from src.pipeline.parser_node import ParserNode
from src.pipeline.verifier_node import VerifierNode
from src.utils.logging_utils import LoggingUtils

# 设置日志
logger = LoggingUtils.setup_logger(
    name="pipeline_builder",
    log_file="logs/pipeline_builder.log"
)

class Pipeline(Node):
    """
    Pipeline 类，用于组合多个节点
    """
    
    def __init__(self, name: str, nodes: Optional[List[Node]] = None):
        """
        初始化 Pipeline
        
        参数:
        - name: Pipeline 名称
        - nodes: 节点列表
        """
        super().__init__(name)
        self.nodes = nodes or []
    
    def add_node(self, node: Node):
        """
        添加节点
        
        参数:
        - node: 要添加的节点
        
        返回:
        - self: 当前 Pipeline，用于链式调用
        """
        self.nodes.append(node)
        return self
    
    def process(self, data: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        处理数据
        
        参数:
        - data: 输入数据
        - **kwargs: 其他参数
        
        返回:
        - processed_data: 处理后的数据
        """
        # 如果没有节点，直接返回输入数据
        if not self.nodes:
            return data
        
        # 依次运行每个节点
        processed_data = data
        for node in self.nodes:
            logger.info(f"运行节点: {node.name}")
            processed_data = node.process(processed_data, **kwargs)
        
        return processed_data

class PipelineBuilder:
    """
    Pipeline 构建器，用于创建 Pipeline
    """
    
    def __init__(self, name: str):
        """
        初始化 Pipeline 构建器
        
        参数:
        - name: Pipeline 名称
        """
        self.pipeline = Pipeline(name)
    
    def add_parser(self, name: str, parser_type: str, model_name: str, task_type: str, 
                  model_params: Optional[Dict[str, Any]] = None, parser_kwargs: Optional[Dict[str, Any]] = None,
                  output_file: Optional[str] = None):
        """
        添加解析器节点
        
        参数:
        - name: 节点名称
        - parser_type: 解析器类型，可选值为"icl"或"ft"
        - model_name: 模型名称或路径
        - task_type: 任务类型，可选值为"combined"、"qp"或"cp"
        - model_params: 模型参数
        - parser_kwargs: 传递给解析器的额外参数
        - output_file: 输出文件路径（可选）
        
        返回:
        - self: 当前构建器，用于链式调用
        """
        parser_node = ParserNode(
            name=name,
            parser_type=parser_type,
            model_name=model_name,
            task_type=task_type,
            model_params=model_params,
            parser_kwargs=parser_kwargs,
            output_file=output_file
        )
        self.pipeline.add_node(parser_node)
        return self
    
    def add_verifier(self, name: str, verifier_type: str, model_name: str, 
                    model_params: Optional[Dict[str, Any]] = None, verifier_kwargs: Optional[Dict[str, Any]] = None,
                    output_file: Optional[str] = None):
        """
        添加验证器节点
        
        参数:
        - name: 节点名称
        - verifier_type: 验证器类型，可选值为"llm"或"z3"
        - model_name: 模型名称或路径
        - model_params: 模型参数
        - verifier_kwargs: 传递给验证器的额外参数
        - output_file: 输出文件路径（可选）
        
        返回:
        - self: 当前构建器，用于链式调用
        """
        verifier_node = VerifierNode(
            name=name,
            verifier_type=verifier_type,
            model_name=model_name,
            model_params=model_params,
            verifier_kwargs=verifier_kwargs,
            output_file=output_file
        )
        self.pipeline.add_node(verifier_node)
        return self
    
    def build(self):
        """
        构建 Pipeline
        
        返回:
        - pipeline: 构建好的 Pipeline
        """
        return self.pipeline
