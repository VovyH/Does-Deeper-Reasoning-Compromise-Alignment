from openai import OpenAI
import time
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

# for gpt
def chatCompletion(model, messages, temperature, retry_times, round_sleep, fail_sleep, api_key, base_url=None):
    if base_url is None:
        client = OpenAI(
            api_key=api_key
            )
    else:
        client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    try:
        response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
    except Exception as e:
        print(e)
        for retry_time in range(retry_times):
            retry_time = retry_time + 1
            print(f"{model} Retry {retry_time}")
            time.sleep(fail_sleep)
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature
                )
                break
            except:
                continue

    model_output = response.choices[0].message.content.strip()
    time.sleep(round_sleep)

    return model_output

def deepseekCompletion(messages):
    # 修改为直接使用SiliconFlow的配置
    client = OpenAI(
        api_key='sk-wrjmrfodxdmizbtjfagnfmetjfsqqcbcjjiwmyflnijgmjwm',
        base_url='https://api.siliconflow.cn/v1'
    )
    
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1",  # 使用完整模型名称
        messages=[
        {"role": "user", "content": messages}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content

def deepseekCompletion_other(messages):
    # 修改为直接使用SiliconFlow的配置
    client = OpenAI(
        api_key='sk-wrjmrfodxdmizbtjfagnfmetjfsqqcbcjjiwmyflnijgmjwm',
        base_url='https://api.siliconflow.cn/v1'
    )
    
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",  # 使用完整模型名称
        messages=[
        messages
        ],
        temperature=0.6
    )
    return response.choices[0].message.content

# for claude
def claudeCompletion(model, max_tokens, temperature, prompt, retry_times, round_sleep, fail_sleep, api_key, base_url=None):
    if base_url is None:
        client = Anthropic(
            api_key=api_key
            )
    else:
        client = Anthropic(
            base_url=base_url,
            auth_token=api_key
            )   
    try:
        completion = client.completions.create(
            model=model,
            max_tokens_to_sample=max_tokens,
            temperature=temperature,
            prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}"
            )
    except Exception as e:
        print(e)
        for retry_time in range(retry_times):
            retry_time = retry_time + 1
            print(f"{model} Retry {retry_time}")
            time.sleep(fail_sleep)
            try:
                completion = client.completions.create(
                model=model,
                max_tokens_to_sample=max_tokens,
                temperature=temperature,
                prompt=prompt
                )
                break
            except:
                continue

    model_output = completion.completion.strip()
    time.sleep(round_sleep)

    return model_output