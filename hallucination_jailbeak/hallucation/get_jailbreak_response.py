import csv
import os
from Jailbreak_Center import get_response_with_deepseek,get_response_with_qwen
from Parameter import Parameter_args
from tqdm import tqdm

def process_csv(input_csv_path, output_csv_path, start_row=371, buffer_size=10):
    """
    读取输入CSV文件，从指定行开始调用模型处理，并缓冲写入结果。

    Args:
        input_csv_path (str): 输入CSV路径
        output_csv_path (str): 输出CSV路径
        start_row (int): 从第 start_row 行开始（0-based，371 表示第372行）
        buffer_size (int): 每处理 buffer_size 条就写入一次
    """
    # 读取所有行（适用于中小规模数据集）
    with open(input_csv_path, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames:
        raise ValueError("Input CSV has no header.")

    if 'response' not in fieldnames:
        fieldnames = fieldnames + ['response']

    total_rows = len(rows)
    if start_row >= total_rows:
        print(f"Start row {start_row} >= total rows {total_rows}. Nothing to process.")
        return

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # 检查输出文件是否已存在（用于决定是否写 header）
    file_exists = os.path.isfile(output_csv_path)
    buffer = []

    # 从 start_row 开始处理
    for i in tqdm(range(start_row, total_rows), desc="Processing Attack Method", total=total_rows - start_row):
        row = rows[i].copy()  # 避免修改原始数据
        prompt = row.get('prompt', '')  # 假设列名为 'prompt'

        try:
            response = get_response_with_qwen(
                question=prompt,
                api_key=Parameter_args.jailbreak_key,
                model=Parameter_args.chat_model_type,
                max_tokens=Parameter_args.max_tokens,
                enable_thinking=Parameter_args.enable_thinking,
                thinking_budget=Parameter_args.thinking_budget,
                temperature=0.6
            )
            row['response'] = response
        except Exception as e:
            row['response'] = f"Error: {str(e)}"

        buffer.append(row)

        # 缓冲区满或到达最后一行时写入
        if len(buffer) >= buffer_size or i == total_rows - 1:
            write_header = not file_exists
            with open(output_csv_path, mode='a', encoding='utf-8', newline='') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerows(buffer)
            file_exists = True  # 后续追加不再写 header
            buffer = []  # 清空缓冲区

    print(f"\n✅ Processing completed! Results saved to: {os.path.abspath(output_csv_path)}")

# ================== 主程序 ==================
if __name__ == "__main__":
    input_csv = r"E:\my_code\LLM_Study\2025ACL_CHJ\CHJ\hallucination_jailbeak\result\jailbreak_result\prompt_result_PAP_Qwen3_thinking.csv"
    output_csv = r"E:\my_code\LLM_Study\2025ACL_CHJ\CHJ\hallucination_jailbeak\result\jailbreak_result\new\result-PAP-Qwen3_thinking.csv"
    
    process_csv(
        input_csv_path=input_csv,
        output_csv_path=output_csv,
        start_row=371,      # 从第372行开始（0-based index 371）
        buffer_size=10      # 每10条保存一次
    )