import re
from .Verifier import Verifier, LLM_VERIFT_SYSTEM_PROMPT

class LLMVerifier(Verifier):
    def __init__(self, model):
        super().__init__(model)
    
    def verify(self, statement, evidence, **kwargs):
        """
        使用LLM验证陈述是否可以从证据中推导出来
        
        参数:
        - statement: 陈述
        - evidence: 证据
        - **kwargs: 其他参数
        
        返回:
        - 验证结果："true" 或 "false"
        """
        # 构建提示
        prompt = f'Statement: "{statement}"\nEvidence: "{evidence}"\n\nCan the statement be deduced from the evidence? Answer with True or False.'
        
        # 设置系统提示
        kwargs["system_prompt"] = LLM_VERIFT_SYSTEM_PROMPT
        
        # 调用模型生成响应
        response = self.model.generate_response(prompt, **kwargs)
        
        # 处理响应
        if not isinstance(response, str):
            return "false"
        
        true_match = re.search(r'\btrue\b', response.lower())
        false_match = re.search(r'\bfalse\b', response.lower())
        
        if true_match and not false_match:
            return "true"
        elif false_match and not true_match:
            return "false"
        elif true_match and false_match:
            # 如果同时包含true和false，可以基于它们的位置或上下文做决定
            # 这里简单地取第一个出现的
            if response.lower().find('true') < response.lower().find('false'):
                return "true"
            else:
                return "false"
        else:
            return "false"