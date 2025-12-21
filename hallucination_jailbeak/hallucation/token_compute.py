import pandas as pd
import tiktoken

def count_tokens_in_response(csv_path: str, encoding_name: str = "cl100k_base") -> pd.Series:
    """
    读取 CSV 文件中的 'response' 列，计算每行文本的 token 数量。

    参数:
        csv_path (str): CSV 文件路径。
        encoding_name (str): tiktoken 使用的编码名称，默认为 "cl100k_base"（适用于 gpt-3.5-turbo、gpt-4 等）。

    返回:
        pd.Series: 每行对应的 token 数量。
    """
    # 读取 CSV
    df = pd.read_csv(csv_path)

    # 检查 'response' 列是否存在
    if 'response' not in df.columns:
        raise ValueError("CSV 文件中缺少 'response' 列。")

    # 加载编码器
    encoder = tiktoken.get_encoding(encoding_name)

    # 计算每行 token 数量（缺失值处理为 0）
    token_counts = df['response'].fillna('').apply(lambda x: len(encoder.encode(str(x))))

    return token_counts
# D:\个人信息\Code\2025ACL_CHJ\2025ACL_CHJ\CHJ\hallucination_jailbeak\result\jailbreak_result\new\evaluate_PAP_llama2_13B_no_reasoning.csv
token_counts = count_tokens_in_response(r"D:\个人信息\Code\2025ACL_CHJ\2025ACL_CHJ\CHJ\hallucination_jailbeak\result\jailbreak_result\NoAttack_responses_advBench_DeepSeek_no_thinking.csv")
print(token_counts.sum())  # 总 token 数
print(token_counts.head()) # 查看前几行