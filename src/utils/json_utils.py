import json
import re
from typing import Dict, Any, Union, Optional


class JsonUtils:
    @staticmethod
    def extract_json_from_text(text: str) -> Union[Dict, str]:
        """
        从文本中提取JSON部分
        
        Args:
            text: 包含JSON的文本
        
        Returns:
            提取的JSON对象或原始文本（如果提取失败）
        """
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # 尝试查找JSON对象的开始和结束
        try:
            # 查找第一个左大括号和最后一个右大括号
            start = text.find('{')
            end = text.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = text[start:end]
                return json.loads(json_str)
        except:
            pass
        
        # 尝试使用正则表达式查找JSON对象
        try:
            json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
            matches = re.findall(json_pattern, text)
            if matches:
                for potential_json in matches:
                    try:
                        return json.loads(potential_json)
                    except:
                        continue
        except:
            pass
        
        return text
    
    @staticmethod
    def format_json(self, data):
        if not data:
            return ""
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    