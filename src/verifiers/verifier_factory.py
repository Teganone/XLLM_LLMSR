from src.verifiers.verifier import Verifier
from src.verifiers.llm_verifier import LLMVerifier
from src.verifiers.z3_verifier import Z3Verifier

class VerifierFactory:
    """验证器工厂，用于创建不同类型的验证器"""
    
    @staticmethod
    def create_verifier(verifier_type, model=None, **kwargs):
        """
        创建验证器
        
        参数:
        - verifier_type: 验证器类型，可以是"llm"或"z3"
        - model: 用于验证的模型
        - **kwargs: 其他参数
        
        返回:
        - 验证器实例
        """
        if verifier_type.lower() == "llm":
            return LLMVerifier(model, **kwargs)
        elif verifier_type.lower() == "z3":
            return Z3Verifier(model, **kwargs)
        else:
            raise ValueError(f"不支持的验证器类型: {verifier_type}")
        


if __name__ == '__main__':
    from src.models.openai_model import OpenaiModel
    from src.verifiers.verifier_factory import VerifierFactory
    import json

    # 加载数据
    with open("results/test.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # 创建模型
    model = OpenaiModel("o3-mini",{'reasoning_effect':'high'})

    # 创建LLM验证器
    llm_verifier = VerifierFactory.create_verifier("llm", model, reasoning_effect='low')

    # # 验证单个陈述
    # result = llm_verifier.verify(
    #     statement="地球是圆的",
    #     evidence="地球是一个近似球体",
    #     question="关于地球的形状"
    # )
    # print(f"LLM验证结果: {result}")

    # 批量验证数据
    # llm_results = llm_verifier.verify(
    #     data=data[:5],
    #     output_file="results/llm_results.json",
    #     batch_size=2,
    #     max_retries=3
    # )

    # 创建Z3验证器
    z3_verifier = VerifierFactory.create_verifier("z3", model)

    # 验证单个陈述
    # result = z3_verifier.verify_statement(
    #     statement="x > 5",
    #     evidence="x = 10",
    #     question="关于x的值"
    # )
    # print(f"Z3验证结果: {result}")

    # 批量验证数据
    z3_results = z3_verifier.verify(
        data=data[:3],
        output_file="results/z3_results.json",
        batch_size=2,
        max_retries=3
    )