from src.parsers.parsing_generator import ParsingGenerator
from src.parsers.icl_parser import ICLParser
from src.parsers.ft_parser import FTParser

class ParserFactory:
    """验证器工厂，用于创建不同类型的验证器"""
    
    @staticmethod
    def create_parser(parser_type, task_type="combined", model=None, **kwargs):
        """
        创建解析器
        
        参数:
        - parser_type: 解析器类型，可以是"icl"或"ft"
        - model: 用于解析的模型
        - **kwargs: 其他参数
        
        返回:
        - 验证器实例
        """
        if parser_type.lower() == "icl":
            return ICLParser(task_type, model)
        elif parser_type.lower() == "ft":
            return FTParser(task_type, model)
        else:
            raise ValueError(f"不支持的解析器类型: {parser_type}")
        


if __name__ == '__main__':
    from src.models.openai_model import OpenaiModel
    from src.models.llama import LlamaModel
    from src.utils.json_utils import JsonUtils
    

    data = JsonUtils.load_json("results/test.json")

    # 创建模型
    model_openai = OpenaiModel("o3-mini",{"reasoning_effect":'low'})
    print(model_openai.get_params())
    model_llama = LlamaModel(model_path="/datacenter/models/LLM-Research/Llama-3-8B-Instruct",params={})
    print(model_llama.get_params())

    # 创建LLM验证器
    iclparser = ParserFactory.create_parser("icl","qp",model_openai)
    data = JsonUtils.load_json('data/Public_Test_A.json')[:3]
    iclparser.parse(data,output_file='results/test_parser_factory_icl_qp.log')

    iclparser = ParserFactory.create_parser("icl","combined",model_openai)
    data = JsonUtils.load_json('data/Public_Test_A.json')[:3]
    iclparser.parse(data,output_file='results/test_parser_factory_icl_combined.log')

    iclparser = ParserFactory.create_parser("icl","combined",model_llama)
    data = JsonUtils.load_json('data/Public_Test_A.json')[:3]
    iclparser.parse(data,output_file='results/test_parser_factory_icl_combined_llam.log')
    
    # model_llama_ft = LlamaModel("/datacenter/chendanchun/models/finetuned/llama3-finetuned_combined/final_model")
    # print(model_llama_ft.get_params())
    # ftparser = ParserFactory.create_parser("ft","combined",model_llama)
    # data = JsonUtils.load_json('data/Public_Test_A.json')[:3]
    # iclparser.parse(data,output_file='results/test_parser_factory_ft_combined_llam.log')


    