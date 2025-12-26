# 读取csv中的内容，提取指定字段的数据，然后给到大模型进行处理将新的输入的数据替换到原来的文件中
import csv
import os
from tqdm import tqdm
from prompt_utils import pap_prompt  # 导入数据生成提示词
from Parameter import Parameter_args  # 导入全局配置
import requests
import time

# 提供的api代码（保持原有代码不变）
def get_response(question: str, api_key: str, model: str, 
                temperature: float):
    """生成响应

    Args:
        question (str): 问题

    Returns:
        dict: 模型生成响应内容
    """
    url = Parameter_args.model_base_url
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "temperature": temperature
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    max_retries = 5
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay * (attempt + 1))
            continue

def process_aime_dataset(input_csv: str, output_csv: str):
    """处理AIME数据集并生成新CSV"""
    results = []
    
    # 读取原始CSV
    with open(input_csv, 'r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        for row in tqdm(reader, desc="Processing AIME questions"):
            original_input = row['input']
            ground_truth = row['output']
            
            # 生成带无关附加信息的新输入（使用prompt_utils中的data_prompt）
            prompt = pap_prompt.format(Input=original_input)
            
            try:
                # 调用模型获取响应
                response = get_response(
                    question=prompt,
                    api_key=Parameter_args.judge_model_key,
                    model=Parameter_args.judge_model,
                    temperature=0.6
                )
                
                # 提取模型生成内容（根据实际响应结构调整）
                new_input = response.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                # 保存结果
                results.append({
                    "input": original_input,
                    "prompt": prompt,
                    "new_input": new_input,
                    "output": ground_truth
                })
            except Exception as e:
                print(f"处理问题时出错，错误: {str(e)}")
                results.append({
                    "input": original_input,
                    "new_input": new_input,
                    "output": ground_truth
                })
    
    # 写入新CSV
    with open(output_csv, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["input", "prompt","new_input", "output"])
        writer.writeheader()
        writer.writerows(results)
    print(f"处理完成，结果保存至: {output_csv}")

if __name__ == "__main__":
    # 配置输入输出路径（使用Parameter中的配置）
    INPUT_CSV = Parameter_args.AIME_PATH  # 从全局配置获取输入路径（对应aime_test.csv）
    OUTPUT_CSV = os.path.join(Parameter_args.backend_dir, "aime_pap.csv")  # 生成新CSV路径
    
    # 执行处理
    process_aime_dataset(INPUT_CSV, OUTPUT_CSV)
