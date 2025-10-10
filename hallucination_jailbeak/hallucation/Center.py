import json
import csv
import time
from tqdm import tqdm
import requests
from Parameter import Parameter_args  # 导入配置参数

def get_response(question: str, api_key: str, model: str, 
                 max_tokens: int, enable_thinking: bool, 
                 thinking_budget: int, temperature: float):
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
        "max_tokens": max_tokens,
        "temperature": temperature,
        "enable_thinking": enable_thinking,
        "thinking_budget": thinking_budget
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

def process_dataset(input_csv: str, output_json: str):
    """处理数据集并保存结果"""
    results = []
    
    # 读取CSV文件
    with open(input_csv, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(tqdm(reader, desc="Processing questions")):
            question = row['new_input']
            ground_truth = row['output']
            try:
                response = get_response(
                    question=question,
                    api_key=Parameter_args.other_key_2,
                    model=Parameter_args.chat_model_type,
                    max_tokens=Parameter_args.max_tokens,
                    enable_thinking=Parameter_args.enable_thinking,
                    thinking_budget=Parameter_args.thinking_budget,
                    temperature=Parameter_args.temperature
                )
                
                # 提取所需内容
                if 'choices' in response and len(response['choices']) > 0:
                    choice = response['choices'][0]
                    result = {
                        "idx": idx,  # 添加正确的索引值
                        "question": question,
                        "content": choice['message'].get('content', ''),
                        "reasoning_content": choice['message'].get('reasoning_content', ''),
                        "ground_truth": ground_truth,
                        "response": response
                    }
                    results.append(result)
            except Exception as e:
                print(f"处理问题时出错，错误: {str(e)}")
    
    # 保存结果到JSON文件
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    INPUT_CSV = Parameter_args.AIME_PATH_METHOD
    
    # 添加循环控制运行次数（从1到run_number）
    for run_count in range(1, Parameter_args.run_count + 1):
        # 动态生成带当前次数的输出文件名
        OUTPUT_JSON = f"{Parameter_args.backend_dir}response_{Parameter_args.model_path}_{Parameter_args.mode}_{Parameter_args.thinking_budget}_{Parameter_args.METHOD}_{run_count}.json"
        
        process_dataset(INPUT_CSV, OUTPUT_JSON)
        print(f"第 {run_count} 次响应处理完成")

    print("所有次数运行完成")