import json
import csv

# 输入文件路径
json_file_path = r"E:\my_code\LLM_Study\2025ACL_CHJ\CHJ\hallucination_jailbeak\result\jailbreak_result\result-FlipAttack-deepseek-V3-advbench.json"
csv_file_path = r"E:\my_code\LLM_Study\2025ACL_CHJ\CHJ\hallucination_jailbeak\result\jailbreak_result\result-FlipAttack-deepseek-V3-advbench.csv"

# 读取JSON文件
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# 打开CSV文件并写入数据
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    # 写入表头
    writer.writerow(["flip_attack", "output"])

    # 写入数据
    for item in data:
        writer.writerow([item.get("flip_attack", ""), item.get("output", "")])

print(f"数据已保存到 {csv_file_path}")