import csv
import json
import os

def insert_ground_truth(csv_path, json_path, new_json_path):  # 新增 new_json_path 参数
    # 读取 CSV 文件的 output 列（按顺序存储）
    output_list = []
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            output_list.append(row['output'].strip())

    # 检查 JSON 文件是否存在
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON 文件不存在：{json_path}")

    # 读取 JSON 文件（错误处理）
    try:
        with open(json_path, 'r', encoding='utf-8') as jsonfile:
            data = json.load(jsonfile)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 文件格式错误：{e}。请检查文件内容是否为有效 JSON。") from e

    # 检查条目数量一致性
    if len(data) != len(output_list):
        raise ValueError(f"CSV 条目数 ({len(output_list)}) 与 JSON 条目数 ({len(data)}) 不一致")

    # 按顺序插入 ground_truth
    for idx, item in enumerate(data):
        item['ground_truth'] = output_list[idx]

    # 写回新文件（使用传入的 new_json_path）
    with open(new_json_path, 'w', encoding='utf-8') as f:  # 避免变量名冲突（原 new_file 改为 f）
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    csv_file = r'e:\我的论文和代码\LLM_Study\hallucination_jailbeak\data/aime_test.csv'
    json_file = r'e:\我的论文和代码\LLM_Study\hallucination_jailbeak\result\results_Qwen3-8B_reasoning.json'
    new_file = r'e:\我的论文和代码\LLM_Study\hallucination_jailbeak\result\results_Qwen3-8B_reasoning_ground_truth.json'
    
    try:
        insert_ground_truth(csv_file, json_file, new_file)  # 传递新文件路径给函数
        print("ground_truth 按顺序插入完成！")
    except Exception as e:
        print(f"执行失败：{str(e)}")