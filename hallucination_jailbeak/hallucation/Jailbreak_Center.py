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
        "Authorization": "Bearer sk-76dqA5HF2oDe7ReFsuhqB8DAEPQA4Cy8uJTlEzFpCkOueQ7o",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o",
        "temperature": 0,
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
        # "thinking_budget": thinking_budget
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
            response_json =  response.json()
            content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay * (attempt + 1))
            continue
        
def get_response_with_gpt(question: str, temperature: float,max_retries=5, retry_delay=2):
    """
    调用 GPT API 并支持自动重试
    
    Args:
        messages: 包含 system 和 user 消息的列表
        max_retries: 最大重试次数，默认 5 次
        retry_delay: 每次重试前的等待时间（秒），默认 2 秒
    """
    url = "https://api.kksj.org/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer sk-zzBVoJToeoS80WNj0wTgZ5MpkV2EJHLqNAnUvBpiYF5bGem8",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-5",
        "temperature": temperature,
        "reasoning_effort": "minimal",
        "messages": [
            {"role": "user", "content": question}
        ]
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            response_json = response.json()

            # 安全提取内容
            choices = response_json.get("choices", [])
            if not choices:
                raise ValueError("No choices in response")
            content = choices[0].get("message", {}).get("content", "")
            if content == "":
                raise ValueError("Empty content in response")
            
            return response_json  # 成功则直接返回

        except (requests.exceptions.RequestException, KeyError, ValueError) as e:
            print(f"[Attempt {attempt}/{max_retries}] Request failed: {e}")
            
            # 如果是最后一次重试，不再等待，直接抛出异常
            if attempt == max_retries:
                print("Max retries reached. Giving up.")
                raise  # 重新抛出最终异常
            else:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)  # 等待后再重试

        
def get_response_with_gpt_reasoning(question: str, temperature: float,max_retries=5, retry_delay=2):
    """
    调用 GPT API 并支持自动重试
    
    Args:
        messages: 包含 system 和 user 消息的列表
        max_retries: 最大重试次数，默认 5 次
        retry_delay: 每次重试前的等待时间（秒），默认 2 秒
    """
    url = "https://api.kksj.org/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer sk-bg7qRHDSvCE30miRNcQtroVCPyUojqivHJufbveTrFKKMSyl",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-5",
        "temperature": temperature,
        "reasoning_effort": "medium",
        "messages": [
            {"role": "user", "content": question}
        ]
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            response_json = response.json()

            # 安全提取内容
            choices = response_json.get("choices", [])
            if not choices:
                raise ValueError("No choices in response")
            content = choices[0].get("message", {}).get("content", "")
            if content == "":
                raise ValueError("Empty content in response")
            
            return response_json  # 成功则直接返回

        except (requests.exceptions.RequestException, KeyError, ValueError) as e:
            print(f"[Attempt {attempt}/{max_retries}] Request failed: {e}")
            
            # 如果是最后一次重试，不再等待，直接抛出异常
            if attempt == max_retries:
                print("Max retries reached. Giving up.")
                raise  # 重新抛出最终异常
            else:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)  # 等待后再重试


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
