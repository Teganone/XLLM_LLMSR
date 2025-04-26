import json
import os

def create_multishot_prompts():
    # 创建输出目录
    os.makedirs("prompts/multishot", exist_ok=True)
    
    # 读取训练数据
    with open("data/Final_Selection_Train_v2.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    
    # 读取现有提示模板
    with open("prompts/extract_combined.txt", "r", encoding="utf-8") as f:
        combined_template = f.read()
    
    with open("prompts/extract_qp.txt", "r", encoding="utf-8") as f:
        qp_template = f.read()
    
    # 提取模板部分（示例之前的部分）
    combined_instructions = combined_template.split("------\nEXAMPLE 1:")[0]
    qp_instructions = qp_template.split("------\nEXAMPLE:")[0]
    
    selected_ids = [162, 374, 208, 375, 441, 476, 107, 412, 188] #given diversity
    selected_samples = [sample for sample in train_data if sample["id"] in selected_ids]
    # 生成combined多示例提示
    combined_examples = []
    for i, sample in enumerate(selected_samples, 1):
        example = f"------\nEXAMPLE {i}:\nquestion:\n{sample['question']}\ncot:\n{sample['cot']}\n\nExample output:\n"
        example += json.dumps({
            "question_parsing": sample["question_parsing"],
            "cot_parsing": sample["cot_parsing"]
        }, ensure_ascii=False, indent=4)
        combined_examples.append(example)
    
    # 生成qp多示例提示
    qp_examples = []
    for i, sample in enumerate(selected_samples, 1):
        example = f"------\nEXAMPLE {i}:\nquestion:\n{sample['question']}\n\nExample output:\n"
        example += json.dumps({
            "question_parsing": sample["question_parsing"]
        }, ensure_ascii=False, indent=4)
        qp_examples.append(example)
    
    # 组合模板和示例
    combined_prompt = combined_instructions + "\n".join(combined_examples) + """

------
NOW ANALYZE THIS NEW QUESTION AND COT:

question:
[[question]]
cot:
[[cot]]

Remember to analyze the specific content above, not the examples. Your output should be a valid JSON object with "question_parsing" and "cot_parsing" keys.
"""
    
    qp_prompt = qp_instructions + "\n".join(qp_examples) + """

------
NOW ANALYZE THIS NEW QUESTION:

question:
[[question]]

Remember to analyze the specific content above, not the examples. Your output should be a valid JSON object with a "question_parsing" key.
"""
    
    # 保存新的提示文件
    with open("prompts/multishot/extract_combined_fewshot.txt", "w", encoding="utf-8") as f:
        f.write(combined_prompt)
    
    with open("prompts/multishot/extract_qp_fewshot.txt", "w", encoding="utf-8") as f:
        f.write(qp_prompt)
    
    print("多示例提示文件已生成：")
    print("- prompts/multishot/extract_combined_fewshot.txt")
    print("- prompts/multishot/extract_qp_fewshot.txt")

if __name__ == "__main__":
    create_multishot_prompts()