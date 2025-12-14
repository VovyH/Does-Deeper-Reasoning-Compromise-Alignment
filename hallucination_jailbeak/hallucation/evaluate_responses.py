import json
from random import choice
import requests
import time
import csv
from tqdm import tqdm
from prompt_utils import evaluate_prompt
from Parameter import Parameter_args  # 导入全局配置


def evaluate_response(question: str, model_response: str, correct_answer: str, api_key: str) -> bool:
    """评估响应

    Args:
        question (str): 问题
        model_response (str): 模型响应
        correct_answer (str): 真值
        api_key (str): 密钥

    Returns:
        bool: True/False
    """
    formatted_prompt = evaluate_prompt.format(
        problem=question,
        model_response=model_response,
        correct_answer=correct_answer
    )
    
    url = Parameter_args.model_base_url  # 使用全局配置中的API地址
    
    payload = {
        "model": Parameter_args.judge_model,  # 使用全局配置中的模型类型
        "messages": [{"role": "user", "content": formatted_prompt}],  # 修正：使用评估提示词作为输入
        "max_tokens": 1024,
        "temperature": 0.0
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
            response_data = response.json()
            # 修正：正确获取choices数组（原代码中拼写错误为'choice'）
            choice = response_data.get('choices', [{}])[0]
            content = choice.get('message', {}).get('content', '')
            return "true" in content.lower() 
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay * (attempt + 1))
            return False



def load_responses(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    # 配置基础路径（从全局配置获取输出目录）
    base_dir = Parameter_args.backend_dir
    # 定义需要评估的文件编号范围（根据用户需求调整）
    run_numbers = range(6,9)  # 对应1-4号文件
    
    for run_number in run_numbers:
        # 动态生成输入输出路径
        input_file = f"{base_dir}response_{Parameter_args.model_path}_{Parameter_args.mode}_{Parameter_args.thinking_budget}_{Parameter_args.METHOD}_{run_number}.json"
        output_file = f"{base_dir}evaluation_{Parameter_args.model_path}_{Parameter_args.mode}_{Parameter_args.thinking_budget}_{Parameter_args.METHOD}_{run_number}.json"
        
        print(f"开始评估文件: {input_file}")
        responses = load_responses(input_file)
        results = []
        
        for item in tqdm(responses, desc=f"评估第 {run_number} 个文件"):
            question = item.get("question")
            content = item.get("content")
            ground_truth = item.get("ground_truth")
            
            if all([question, content, ground_truth]):
                is_correct = evaluate_response(
                    question=question,
                    model_response=content,
                    correct_answer=ground_truth,
                    api_key=Parameter_args.motivation_key  # 使用全局配置中的API密钥
                )
                results.append({
                    "question": question,
                    "model_response": content,
                    "ground_truth": ground_truth,
                    "is_correct": is_correct
                })
        
        # 保存当前文件的评估结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"评估完成，结果保存至: {output_file}\n")

if __name__ == "__main__":
    main()