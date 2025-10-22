import json
import csv
import time
from tqdm import tqdm
import requests
from Parameter import Parameter_args  # 导入配置参数
from openai import OpenAI
    
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 可选：定义哪些异常需要重试
@retry(
    stop=stop_after_attempt(3),  # 最多重试 3 次
    wait=wait_exponential(multiplier=1, min=2, max=10),  # 指数退避：2s, 4s, 8s...
    retry=retry_if_exception_type((requests.exceptions.RequestException, KeyError, ValueError))
)
def get_response_with_judge_gpt(question):
    url = "https://api.kksj.org/v1/chat/completions"
    headers = {
        "Authorization": "Bearer sk-zzBVoJToeoS80WNj0wTgZ5MpkV2EJHLqNAnUvBpiYF5bGem8",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()  # 如果状态码不是 2xx，会抛出 HTTPError
        response_json = response.json()

        # 安全提取内容
        choices = response_json.get("choices", [])
        if not choices:
            raise ValueError("No choices in response")
        content = choices[0].get("message", {}).get("content", "")
        if content == "":
            raise ValueError("Empty content in response")
        return content

    except (requests.exceptions.RequestException, KeyError, ValueError) as e:
        print(f"Request failed: {e}")
        raise  # 重新抛出异常以触发重试



def get_response_with_qwen(question: str, api_key: str, model: str, 
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


def get_response_with_judge(question: str, api_key: str, 
                 max_tokens: int,  
                temperature: float):
    """生成响应

    Args:
        question (str): 问题

    Returns:
        dict: 模型生成响应内容
    """
    url = Parameter_args.model_base_url
    
    payload = {
        "model": Parameter_args.judge_model,
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_tokens,
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
            response_json = response.json()
            content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")

            return content
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay * (attempt + 1))
            continue


def get_response_with_deepseek(question: str, api_key: str, model: str, 
                 max_tokens: int,  
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
        "max_tokens": max_tokens,
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
            response_json = response.json()
            content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")

            return content
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay * (attempt + 1))
            continue
