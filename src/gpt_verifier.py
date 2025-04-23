import os
import sys
# 将当前目录的绝对路径插入到sys.path的最前面
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_dir)

import json
import re
from tqdm import tqdm
import argparse
from openai_api import get_validation_response, MODEL

import json


def process_validation_inference(test_data, output_file, model, batch_size = 10):
    results = []
        
    prompt = 'Whether the "statement" can be deduced from the "evidence" logically, answer with only with True or False, do not output other contents.'
    for item in tqdm(test_data, desc="处理测试数据"):
        id = item['id']
        text_pre = 'question: ' + item['question'] + "\n"
        for sev in item['cot_parsing']:
            try:
                text = text_pre + 'statement: ' + sev['statement'] + ' ' + 'evidence: ' + sev['evidence']
                response = get_validation_response(prompt, text, model=model)
            
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
                        sev['Verification'] = "false"
                else:
                    print("answer not true or false:", response)
                    if 'Verification' not in sev or sev['Verification'] not in ['true','True','False','false']:
                        sev['Verification'] = "false"
            except:
                pass
                
                
        result = {
            "question": item['question'],
            "question_parsing": item["question_parsing"],
            "answer": item["answer"],
            "id": item["id"],
            "cot": item["cot"],
            "cot_parsing": item["cot_parsing"],
            # "sel_idx":item['sel_idx']
        }
        results.append(result)
        if len(results) % batch_size == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"已保存{len(results)}个结果到{output_file}")
    
    # 保存最终结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成，共{len(results)}个结果已保存到{output_file}")




def main(args):
    with open(args.input, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    process_validation_inference(test_data=raw_data,output_file=args.output, model=args.model)



def parse_args():
    parser = argparse.ArgumentParser(description="Verification of GPT for LLMSR")
    # parser.add_argument("--input", type=str, default="LLMSR_Datasets/Output Results/Public_Test_result_v5_Llama-3-8B-Instruct-icl.json", help="Input JSON file")
    parser.add_argument('--input',type=str, required=True, help="Input JSON file")
    # parser.add_argument("--output", type=str, default="LLMSR_Datasets/Output Results/Verified_Public_Test_result_v5_Llama-3-8B-Instruct-icl.json", help="Output JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("--model", type=str, required=True, help="model")


    return parser.parse_args()
    
    


if __name__ == '__main__':
    args = parse_args()
    main(args)
   