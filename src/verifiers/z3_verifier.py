from .Verifier import Verifier
from .z3_solver.smt_solver import LLMSR_Z3_Program
import re

class Z3Verifier(Verifier):
    def __init__(self):
        """
        初始化Z3Verifier
        """
        super().__init__()
    
    def verify(self, statement, evidence, **kwargs):
        """
        使用Z3求解器验证陈述是否可以从证据中推导出来
        
        参数:
        - statement: 陈述
        - evidence: 证据
        - **kwargs: 其他参数
        
        返回:
        - 验证结果："true" 或 "false"
        """
        # 构建逻辑程序
        logic_program = self._build_logic_program(statement, evidence)
        
        # 创建Z3程序
        z3_program = LLMSR_Z3_Program(logic_program)
        
        # 如果程序初始化失败，返回false
        if not z3_program.flag:
            return "false"
        
        # 执行程序
        output, error_message = z3_program.execute_program()
        
        # 处理输出
        if output is None or error_message:
            return "false"
        
        # 解析输出
        try:
            # 输出格式应该是 "All verification results: [True]" 或 "All verification results: [False]"
            result_match = re.search(r'\[(True|False)\]', output[0])
            if result_match:
                result = result_match.group(1)
                return result.lower()
            else:
                return "false"
        except Exception as e:
            print(f"解析Z3输出失败: {e}")
            return "false"
    
    def _build_logic_program(self, statement, evidence):
        """
        构建逻辑程序
        
        参数:
        - statement: 陈述
        - evidence: 证据
        
        返回:
        - 逻辑程序字符串
        """
        # 构建一个简单的逻辑程序模板
        logic_program = f"""
# Declarations
bool_sort = EnumSort([True, False])

# Constraints
{evidence} ::: Evidence

# Verifications
is_deduced({evidence}, {statement}) ::: Can the statement be deduced from the evidence?
"""
        return logic_program