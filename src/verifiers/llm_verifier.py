import re
from src.verifiers.verifier import Verifier
from tqdm import tqdm
from src.utils.json_utils import JsonUtils

LLM_VERIFT_SYSTEM_PROMPT = 'Whether the "statement" can be deduced from the "evidence" logically, answer with only with True or False, do not output other contents.'

class LLMVerifier(Verifier):
    def __init__(self, model, **kwargs):
        """
        初始化LLM验证器
        
        参数:
        - model: 用于验证的模型
        """
        super().__init__(model,**kwargs)
    
    def load_prompt_templates(self):
        self.system_prompt = LLM_VERIFT_SYSTEM_PROMPT


    def set_messages(self, user_prompt):
        messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        return messages



    def verify(self, data, output_file, batch_size=10, **kwargs):
        results = []
        for item in tqdm(data, desc="处理测试数据"):
            prompt_pre = 'question: ' + item['question'] + "\n"
            for sev in item['cot_parsing']:
                try:
                    user_prompt = prompt_pre + 'statement: ' + sev['statement'] + ' ' + 'evidence: ' + sev['evidence']
                    messages = self.set_messages(user_prompt)
                    response = self.model.invoke(messages, **kwargs)
                    if not isinstance(response, str):
                        print("response not str:",response)
                        if 'Verification' not in sev or sev['Verification'] not in ['true','True','False','false']:
                            sev['Verification'] = "false"
                        continue
                    true_match = re.search(r'\btrue\b', response.lower())
                    false_match = re.search(r'\bfalse\b', response.lower())
                    if true_match and not false_match:
                        sev['Verification'] = "true"
                    elif false_match and not true_match:
                        sev['Verification'] = "false"
                    elif true_match and false_match:
                        # 如果同时包含true和false，可以基于它们的位置或上下文做决定
                        # 这里简单地取第一个出现的
                        if response.lower().find('true') < response.lower().find('false'):
                            sev['Verification'] = "true"
                        else:
                            sev['Verification'] = "true"
                    else:
                        print("answer not true or false:", response)
                        if 'Verification' not in sev or sev['Verification'] not in ['true','True','False','false']:
                            sev['Verification'] = "true"
                except:
                    pass
                    
                    
            result = {
                "question": item['question'],
                "question_parsing": item["question_parsing"],
                "answer": item["answer"],
                "id": item["id"],
                "cot": item["cot"],
                "cot_parsing": item["cot_parsing"],
            }
            results.append(result)
            if output_file and len(results) % batch_size == 0:
                JsonUtils.save_to_file(results, output_file)
                print(f"已保存{len(results)}个结果到{output_file}")
        if output_file:
            JsonUtils.save_to_file(results, output_file)
            print(f"处理完成，共{len(results)}个结果已保存到{output_file}")

        return results
