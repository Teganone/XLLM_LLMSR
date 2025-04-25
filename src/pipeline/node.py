from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

class Node(ABC):
    """
    Pipeline 节点基类
    """
    
    def __init__(self, name: str):
        """
        初始化节点
        
        参数:
        - name: 节点名称
        """
        self.name = name
        self.next_nodes = []
    
    def add_next(self, node: 'Node'):
        """
        添加下一个节点
        
        参数:
        - node: 下一个节点
        
        返回:
        - self: 当前节点，用于链式调用
        """
        self.next_nodes.append(node)
        return self
    
    @abstractmethod
    def process(self, data: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        处理数据
        
        参数:
        - data: 输入数据
        - **kwargs: 其他参数
        
        返回:
        - processed_data: 处理后的数据
        """
        pass
    
    def run(self, data: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        运行节点及其后续节点
        
        参数:
        - data: 输入数据
        - **kwargs: 其他参数
        
        返回:
        - result: 处理结果
        """
        # 处理当前节点
        processed_data = self.process(data, **kwargs)
        
        # 如果没有后续节点，返回处理结果
        if not self.next_nodes:
            return processed_data
        
        # 否则，运行后续节点
        for node in self.next_nodes:
            processed_data = node.run(processed_data, **kwargs)
        
        return processed_data
