import requests
import pandas as pd
import time
import os
from typing import Tuple, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

# ==================== 主调用函数（使用 @retry 装饰器） ====================

@retry(
    stop=stop_after_attempt(5),  # 建议改为 5 次，更稳健
    wait=wait_exponential(multiplier=1, min=2, max=30),  # 最长等 30s
    retry=retry_if_exception_type((
        requests.exceptions.RequestException,
        KeyError,
        ValueError,
        requests.exceptions.Timeout
    )),
    reraise=True  # 最终失败时抛出原始异常
)
def get_response_with_claude(
    question: str,
    enable_thinking: bool = False,
    thinking_budget: Optional[int] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = 8192,
) -> Tuple[str, str]:
    url = "https://api.kksj.org/v1/chat/completions"
    headers = {
        "Authorization": "Bearer sk-76dqA5HF2oDe7ReFsuhqB8DAEPQA4Cy8uJTlEzFpCkOueQ7o",
        "Content-Type": "application/json"
    }

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question}
    ]

    data = {
        "model": "claude-haiku-4-5-20251001",
        "temperature": temperature,
        "messages": messages,
    }

    if max_tokens is not None:
        data["max_tokens"] = max_tokens

    if enable_thinking:
        thinking_config = {"type": "enabled"}
        if thinking_budget is not None:
            if thinking_budget < 1024:
                raise ValueError("thinking_budget 最小值为 1024")
            thinking_config["budget_tokens"] = thinking_budget
        data["thinking"] = thinking_config
    else:
        data["thinking"] = {"type": "disabled"}

    response = requests.post(url, headers=headers, json=data, timeout=300)
    response.raise_for_status()
    response_json = response.json()

    choices = response_json.get("choices", [])
    if not choices:
        raise ValueError("No choices in response")

    message = choices[0].get("message", {})
    content = message.get("content", "")

    think = ""
    final_response = ""

    if isinstance(content, list):  # 支持原生 thinking blocks
        for block in content:
            if block.get("type") == "thinking":
                think += block.get("thinking", "") + "\n"
            elif block.get("type") == "text":
                final_response += block.get("text", "")
        think = think.strip()
        final_response = final_response.strip()
    elif isinstance(content, str):
        final_response = content.strip()

    if not final_response:
        final_response = str(content).strip()

    return think, final_response


# ==================== 批量处理（tqdm + tenacity） ====================

def process_dataset(csv_path: str, output_path: str, enable_thinking: bool, thinking_budget: Optional[int] = None):
    # 文件是否存在的判断
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"文件不存在: {csv_path}")

    # 文件某列的内容是否存在的判断
    df = pd.read_csv(csv_path)
    if 'input' not in df.columns:
        raise ValueError("CSV 必须包含 'input' 列")

    # 结果
    results = []

    # 处理核心逻辑
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=os.path.basename(output_path)):
        question = row['input']
        try:
            think, response = get_response_with_claude(
                question=question,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget,
                temperature=0.0,
                max_tokens=8192
            )
        except Exception as e:
            print(f"\n[{idx+1}] 最终失败: {e}")
            think = "ERROR: Final retry failed"
            response = f"ERROR: {str(e)}"

        results.append({
            "input": question,
            "think": think,
            "response": response
        })

        # 可选：每处理几条保存一次中间结果（防止中断丢失）
        if (idx + 1) % 5 == 0:
            pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8-sig")

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {output_path}")


if __name__ == "__main__":
    import time

    # ==================== 配置区 ====================
    DELAY_BETWEEN_CALLS = 0.6          # 单次 API 调用后的延迟（防限流）
    DELAY_BETWEEN_RUNS = 5             # 每次完整实验（一次保存）后的休息时间
    DELAY_BETWEEN_EXPERIMENTS = 20      # 无思考 和 有思考 两大实验之间的休息时间
    REPEAT_TIMES = 4                   # 每种实验重复次数

    INPUT_CSV = r"D:\个人信息\Code\2025ACL_CHJ\2025ACL_CHJ\CHJ\hallucination_jailbeak\data\aime_feihua.csv"

    # 基础输出路径（不含后缀 _1 _2 等）
    BASE_NO_THINKING = r"D:\个人信息\Code\2025ACL_CHJ\2025ACL_CHJ\CHJ\hallucination_jailbeak\result\aime\feihua\response_feihua_aime_no_thinking_claude"
    BASE_WITH_THINKING_4096 = r"D:\个人信息\Code\2025ACL_CHJ\2025ACL_CHJ\CHJ\hallucination_jailbeak\result\aime\feihua\response_feihua_aime_thinking_4096_claude"
    BASE_WITH_THINKING_2048 = r"D:\个人信息\Code\2025ACL_CHJ\2025ACL_CHJ\CHJ\hallucination_jailbeak\result\aime\feihua\response_feihua_aime_thinking_2048_claude"

    # ================================================

    # 第一大类：无扩展推理，重复 REPEAT_TIMES 次
    print(f"=== 开始无扩展推理实验（重复 {REPEAT_TIMES} 次） ===")
    for i in range(1, REPEAT_TIMES + 1):
        output_path = f"{BASE_NO_THINKING}_{i}.csv"
        print(f"\n--- 第 {i}/{REPEAT_TIMES} 次：保存到 {output_path} ---")
        process_dataset(
            csv_path=INPUT_CSV,
            output_path=output_path,
            enable_thinking=False,
            thinking_budget=None
        )
        print(f"第 {i} 次无思考实验完成。")
        if i < REPEAT_TIMES:
            print(f"休息 {DELAY_BETWEEN_RUNS} 秒后继续下一次...")
            time.sleep(DELAY_BETWEEN_RUNS)

    # 两大实验之间多休息一会儿
    print(f"\n无思考实验全部完成，休息 {DELAY_BETWEEN_EXPERIMENTS} 秒后开始有思考实验...")
    time.sleep(DELAY_BETWEEN_EXPERIMENTS)

    # 第二大类：扩展推理 budget=4096，重复 REPEAT_TIMES 次
    print(f"=== 开始扩展推理实验（budget=4096，重复 {REPEAT_TIMES} 次） ===")
    for i in range(1, REPEAT_TIMES + 1):
        output_path = f"{BASE_WITH_THINKING_4096}_{i}.csv"
        print(f"\n--- 第 {i}/{REPEAT_TIMES} 次：保存到 {output_path} ---")
        process_dataset(
            csv_path=INPUT_CSV,
            output_path=output_path,
            enable_thinking=True,
            thinking_budget=4096
        )
        print(f"第 {i} 次有思考实验完成。")
        if i < REPEAT_TIMES:
            print(f"休息 {DELAY_BETWEEN_RUNS} 秒后继续下一次...")
            time.sleep(DELAY_BETWEEN_RUNS)

    # 第三大类：扩展推理 budget=2048，重复 REPEAT_TIMES 次
    print(f"=== 开始扩展推理实验（budget=2048，重复 {REPEAT_TIMES} 次） ===")
    for i in range(1, REPEAT_TIMES + 1):
        output_path = f"{BASE_WITH_THINKING_2048}_{i}.csv"
        print(f"\n--- 第 {i}/{REPEAT_TIMES} 次：保存到 {output_path} ---")
        process_dataset(
            csv_path=INPUT_CSV,
            output_path=output_path,
            enable_thinking=True,
            thinking_budget=2048
        )
        print(f"第 {i} 次有思考实验完成。")
        if i < REPEAT_TIMES:
            print(f"休息 {DELAY_BETWEEN_RUNS} 秒后继续下一次...")
            time.sleep(DELAY_BETWEEN_RUNS)

    print("\n所有实验全部完成！共生成以下文件：")
    for i in range(1, REPEAT_TIMES + 1):
        print(f"  - response_feihua_aime_no_thinking_claude_{i}.csv")
        print(f"  - response_feihua_aime_thinking_4096_claude_{i}.csv")
        print(f"  - response_feihua_aime_thinking_2048_claude_{i}.csv")