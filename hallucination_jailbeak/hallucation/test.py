import requests
from Parameter import Parameter_args  # 导入配置参数

url = Parameter_args.model_base_url

payload = {
    "model": Parameter_args.chat_model_type,
    "messages": [
        {
            "role": "user",
            "content": "What opportunities and challenges will the Chinese large model industry face in 2025?"
        }
    ],
    "max_tokens": Parameter_args.max_tokens,
    "temperature": Parameter_args.temperature,
    "enable_thinking": Parameter_args.enable_thinking,
    "thinking_budget": Parameter_args.thinking_budget
}
headers = {
    "Authorization": f"Bearer {Parameter_args.model_key}",
    "Content-Type": "application/json"
}

response = requests.request("POST", url, json=payload, headers=headers)

# 新增：检查 HTTP 请求是否成功
if response.status_code != 200:
    print(f"请求失败，状态码：{response.status_code}，原始响应：{response.text}")
    exit(1)

try:
    response_json = response.json()
    # 新增：安全获取 choices 字段，避免 KeyError
    choices = response_json.get('choices', [])
    if not choices:
        print(f"响应中未找到 'choices' 字段，完整响应：{response_json}")
        content = ''
    else:
        content = choices[0].get('message', {}).get('content', '')
except Exception as e:
    print(f"解析响应失败，错误：{str(e)}，原始响应：{response.text}")
    content = ''

print(content)