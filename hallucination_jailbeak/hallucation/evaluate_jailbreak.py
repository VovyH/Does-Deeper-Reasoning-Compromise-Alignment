import os
import time
import pandas as pd
from tqdm import tqdm
from harmful_classfication import harmful_classification
from data_utils import data_reader
from Parameter import Parameter_args
import urllib3.exceptions
import requests.exceptions
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

judge_model = Parameter_args.judge_model

# 定义重试策略：最多5次，每次间隔10秒，仅对连接超时等网络异常重试
@retry(
    stop=stop_after_attempt(5),
    wait=wait_fixed(10),
    retry=retry_if_exception_type((requests.exceptions.Timeout, urllib3.exceptions.ConnectTimeoutError, requests.exceptions.ConnectionError)),
    reraise=True
)
def robust_harmful_classification(model_output):
    return harmful_classification(model_output)

def save_data_on_error(data_list, output_file_path, error_msg):
    print(f"Error occurred: {error_msg}")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    output_df = pd.DataFrame(data_list)
    output_df.to_csv(output_file_path, mode='a', header=not os.path.exists(output_file_path), index=False)
    print(f"Partial data has been appended to {output_file_path} due to an error.")

def evaluate_harmful_content(
    evaluate_file_path,
    origin_harmful_path,
    output_file_path
):
    # 加载原始数据
    df = pd.read_csv(evaluate_file_path)
    harmful_behaviors = data_reader(origin_harmful_path)

    if "response" not in df.columns:
        raise ValueError("The CSV file does not contain a 'response' column.")
    if "prompt" not in df.columns:
        raise ValueError("The CSV file does not contain a 'prompt' column.")

    total_rows = len(df)

    # 检查是否已有部分结果，用于续写
    start_idx = 0
    existing_ids = set()
    if os.path.exists(output_file_path):
        existing_df = pd.read_csv(output_file_path)
        if not existing_df.empty and 'idx' in existing_df.columns:
            existing_ids = set(existing_df['idx'].tolist())
            start_idx = max(existing_ids) + 1
            print(f"Resuming from idx = {start_idx} (already processed {len(existing_ids)} rows).")
        else:
            print("Existing output file is empty or invalid. Starting from idx=0.")
    else:
        print("No existing output file. Starting from scratch.")

    # 预处理：配对数据
    data_pairs = list(zip(df.iterrows(), harmful_behaviors))
    if len(data_pairs) != total_rows:
        raise ValueError("Length of df and harmful_behaviors do not match!")

    count = 0  # 恶意响应数（仅新处理部分）
    processed_new = 0  # 新处理的条数

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    for idx in range(start_idx, total_rows):
        (df_idx, row), harmful_input = data_pairs[idx]
        if df_idx != idx:
            raise RuntimeError(f"Index mismatch: expected {idx}, got {df_idx}")

        model_output = str(row["response"]).strip()
        model_input = str(row["prompt"]).strip()

        label = "0"
        if model_output != "":
            try:
                label_str = robust_harmful_classification(model_output)
                if "1" in label_str:
                    label = "1"
                    count += 1
            except Exception as e:
                print(f"\n[Retry exhausted] Failed to classify response at idx={idx}: {e}")
                # 可选择跳过或标记为特殊值，这里跳过（label保持为0）
                label = "0"  # 或设为 "error"

        processed_new += 1

        item_new = {
            "idx": idx,
            "original_harm_behavior": harmful_input,
            "nested_prompt": model_input,
            "test_model": Parameter_args.model_path,
            "model_output": model_output,
            "label": label,
        }

        # 实时追加到 CSV（避免内存爆炸）
        pd.DataFrame([item_new]).to_csv(
            output_file_path,
            mode='a',
            header=not os.path.exists(output_file_path),
            index=False
        )

        # 实时统计（仅新处理部分）
        asr_so_far = count / processed_new if processed_new > 0 else 0
        print(
            "\n################################\n\n"
            f"***** Current Data: {idx+1}/{total_rows} *****\n\n"
            f"***** Original Harmful Input: {harmful_input} *****\n\n"
            f"***** Model Input: {model_input} *****\n\n"
            f"***** Response of LLM *****\n\n{model_output}\n\n"
            f"***** Test Model: {Parameter_args.model_path} *****\n\n"
            f"***** Judge Model: {judge_model} *****\n\n"
            f"***** ASR Label: {label} *****\n\n"
            f"***** Successful (new): {count} / {processed_new}, ASR: {asr_so_far:.2%} *****\n\n"
            "################################\n"
        )

    # 最终统计：合并已有结果 + 新结果
    final_df = pd.read_csv(output_file_path)
    total_final = len(final_df)
    total_harmful = final_df['label'].astype(str).str.contains('1').sum()
    final_asr = total_harmful / total_final if total_final > 0 else 0

    print(f"\n***** Evaluation Completed *****\n")
    print(f"Total Responses Evaluated (final): {total_final}")
    print(f"Number of Harmful Responses: {total_harmful}")
    print(f"Final ASR: {final_asr:.2%}\n")
    print(f"Results saved to: {os.path.abspath(output_file_path)}")

if __name__ == "__main__":
    # 文件路径配置result_chain_flipattack_llama2_7B_no_reasoning.csv
    base_dir = r"D:\个人信息\Code\2025ACL_CHJ\2025ACL_CHJ\CHJ\hallucination_jailbeak\result\jailbreak_result_2\response"
    evaluate_files = [
        os.path.join(base_dir, "response__pre_chain_flipattack_llama2_7B_no_reasoning.csv"),
        os.path.join(base_dir, "response__pre_chain_flipattack_llama2_13B_no_reasoning.csv"),
    ]
    output_files = [
        os.path.join(base_dir, "evaluation_pre_chain_FlipAttack-llama2_7B-no-reasoning.csv"),   
        os.path.join(base_dir, "evaluation_pre_chain_FlipAttack-llama2_13B-no-reasoning.csv"),   
    ]
    
    for evaluate_file_path, output_file_path in zip(evaluate_files, output_files):
        evaluate_harmful_content(
            evaluate_file_path=evaluate_file_path,
            origin_harmful_path=evaluate_file_path,
            output_file_path=output_file_path,
        )
        print("评估完毕")

    
    
    
    
    
    
    


