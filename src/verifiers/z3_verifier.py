from src.verifiers.verifier import Verifier
from .z3_solver.smt_solver import LLMSR_Z3_Program
import re
from tqdm import tqdm
import time
from src.utils.logging_utils import LoggingUtils
from src.utils.json_utils import JsonUtils
import os

# 设置日志
logger = LoggingUtils.setup_logger(
    name="z3_verifier",
    log_file="logs/z3_verifier.log"
    )

class Z3Verifier(Verifier):
    def __init__(self, model, **kwargs):
        """
        初始化Z3Verifier
        """
        super().__init__(model, **kwargs)
        self.logic_program_dir = "logic_programs"
        self.python_program_dir = "python_programs"
        os.makedirs(self.logic_program_dir, exist_ok=True)
        os.makedirs(self.python_program_dir, exist_ok=True)
        self.load_prompt_templates()

    def load_prompt_templates(self):
        with open("prompts/solver_verification.txt", 'r') as f:
            self.prompt_template = f.read()

    def _resolve_prompt(self, data):
        question = data['question']
        import json
        cot_parsing = json.dumps(data['cot_parsing'], indent=4, ensure_ascii=False)
        full_prompt = self.prompt_template.replace('[[QUESTION]]', question).replace('[[COT_PARSING]]', cot_parsing)
        return full_prompt

    def _get_logic_program(self,prompt):
        """
        使用Azure OpenAI API获取逻辑程序响应
        
        Args:
            prompt: 提示文本
            model: 模型名称
            temperature: 温度参数
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
        
        Returns:
            生成的逻辑程序
        """
        messages=[
                    {"role": "user", "content": prompt},
                ]
        max_retries = 3
        retry_delay = 2
        for attempt in range(max_retries):
            try:
                response = self.model.invoke(messages,max_retries=1)
                logger.info("-" * 50)
                logger.info(f"response:{response}")
                logger.info("-" * 50)
                if isinstance(response, str):
                    # 尝试提取逻辑程序部分
                    content = response
                    if "###" in content:
                        logic_program = content.split("###")[1].strip()
                    elif "# Declarations" in content:
                        logic_program = content[content.find("# Declarations"):].strip()
                    else:
                        logic_program = content
                    return logic_program
                else:
                    logger.info("模型返回空内容")
                    return ""
            except Exception as e:
                logger.info(f"API调用失败 (尝试 {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logger.info("达到最大重试次数，返回空结果")
                    return ""

    def _execute_logic_program(self, logic_program, debug=False):
        """
        执行逻辑程序并获取验证结果
        
        Args:
            logic_program: 逻辑程序
            debug: 是否打印调试信息
            
        Returns:
            验证结果列表和错误信息
        """
        try:
            if debug:
                logger.info("执行逻辑程序:")
                logger.info("-" * 50)
                logger.info(logic_program)
                logger.info("-" * 50)
            
            z3_program = LLMSR_Z3_Program(logic_program)

            output, error_message = z3_program.execute_program()
            
            if debug:
                logger.info("执行结果:")
                logger.info("-" * 50)
                for line in output:
                    logger.info(line)
                logger.info("-" * 50)
                if error_message:
                    logger.info(f"错误信息: {error_message}")
            
            return output, error_message, z3_program.standard_code
        except Exception as e:
            # if debug:
            logger.info(f"执行逻辑程序时出错: {str(e)}")
            return [], str(e), z3_program.standard_code



    def _extract_verification_results(self, output):
        """
        从输出中提取验证结果列表
        
        Args:
            output: 执行逻辑程序的输出
            
        Returns:
            验证结果列表（布尔值）
        """
        for line in output:
            if "All verification results:" in line:
                import re
                match = re.search(r'\[(.*?)\]', line)
                if match:
                    results_str = match.group(1)
                    verification_results = [r.strip() for r in results_str.split(', ')]
                    verification_results = [result.lower() for result in verification_results]
                    return verification_results
        return []  # 如果没有找到结果，返回空列表


    def verify(self, data, output_file, batch_size=10, **kwargs):
        results = []
        for item in tqdm(data, desc="处理测试数据"):
            user_prompt = self._resolve_prompt(item) 
            logic_program = self._get_logic_program(user_prompt)
                
            max_retries = 3
            for retry in range(max_retries):
                output, error_message, standard_code = self._execute_logic_program(logic_program, debug=(retry==0))
                if not error_message or retry == max_retries - 1:
                    break
                
                # 构建修复提示，包含错误信息和之前的逻辑程序
                repair_prompt = f"""
                    I tried to execute the following logic program but encountered an error:

                    ```
                    {logic_program}
                    ```

                    Error message:
                    {error_message}

                    Please fix this logic program according to the error message. Make sure the fixed program follows these rules:
                    1. Use Python syntax instead of mathematical symbols (use 'and' instead of '∧', 'or' instead of '∨', 'not' instead of '¬')
                    2. Ensure all variables are defined
                    3. Don't use comparison operators (<, >, <=, >=) on enumeration types unless the function returns an integer type
                    4. Use == for equality comparison, not Equals function
                    5. Ensure all custom functions are defined in the declarations section

                    Original problem:
                    {user_prompt}

                    Please provide the complete fixed logic program:
                    """
                logger.info(f"尝试第{retry+1}次修复逻辑程序...")
                # 获取修复后的逻辑程序
                logic_program = self._get_logic_program(repair_prompt)
            
            # 更新验证结果
            cot_parsing = item['cot_parsing']
            if not error_message and len(output) > 0:
                # 提取验证结果
                verification_results = self._extract_verification_results(output)
                logger.info(f"提取的验证结果: {verification_results}")
                
                # 更新cot_parsing中的验证结果
                for i, verification in enumerate(verification_results):
                    if i < len(cot_parsing):
                        cot_parsing[i]['Verification'] = verification
            else:
                logger.info(f"执行逻辑程序时出错: {error_message}")
                logger.info("-" * 50)
                logger.info(item['id'])
                logger.info("-" * 50)
            
            # 构建结果
            result = item.copy()
            result['cot_parsing'] = cot_parsing

            with open(f"{self.logic_program_dir}/{item['id']}.txt", 'w', encoding='utf-8') as f:
                f.write(logic_program)
        
            if standard_code is not None:
                with open(f"{self.python_program_dir}/{item['id']}.py", 'w', encoding='utf-8') as f:
                    f.write(standard_code)
            
            results.append(result)
            
            if len(results) % batch_size == 0:
                JsonUtils.save_to_file(results, output_file)
                logger.info(f"已保存{len(results)}个结果到{output_file}")
        
        JsonUtils.save_to_file(results, output_file)
        logger.info(f"处理完成，共{len(results)}个结果已保存到{output_file}")