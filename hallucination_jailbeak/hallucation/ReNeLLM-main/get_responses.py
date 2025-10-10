import os
import json
import argparse

from utils.llm_completion_utils import chatCompletion, claudeCompletion
from utils.data_utils import data_reader
from utils.llm_responses_utils import gpt_responses, claude_responses, mistral_responses,deepseek_response
from transformers import AutoModelForCausalLM, AutoTokenizer

def main(args):
    # 读取CSV文件中的有害行为数据
    data = data_reader(args.data_path)
    
    # 只取前50条数据
    data = data[:50]
    
    # 初始化存储结果的数据列表
    data_list = []
    
    # 遍历数据集中的每个条目
    for idx, harm_behavior in enumerate(data):
        model_output = ""
        # 根据不同的测试模型调用相应的响应函数
        if "gpt" in args.test_model:
            model_output = gpt_responses(args, harm_behavior)
        elif "claude" in args.test_model:
            model_output = claude_responses(args, harm_behavior)
        elif "deepseek" in args.test_model:
            model_output = deepseek_response(args, harm_behavior)

        # 打印当前处理的数据信息
        print(
            "\n################################\n\n"
            f"***** Current Data: {idx+1}/{len(data)} *****\n\n"
            f"***** Harm Behavior *****\n{harm_behavior}\n\n"
            f"***** Test Model *****\n{args.test_model}\n\n"
            f"***** Response of {args.test_model} *****\n\n"
            f"{model_output}\n\n"
            "################################\n"
        )

        # 构建新的数据项
        item_new = {
            'idx': idx,
            'original_harm_behavior': harm_behavior,
            'test_model': args.test_model,
            'model_output': model_output
        }
        data_list.append(item_new)

    # 确保保存结果的目录存在
    if not os.path.exists('./results/responses'):
        os.makedirs('./results/responses')

    # 构建保存结果文件的路径 - 改为CSV格式
    file_name = f"./results/responses/responses_of_on_{args.test_model}_{args.save_suffix}.csv"

    # 将结果列表保存为CSV文件
    import pandas as pd
    df = pd.DataFrame(data_list)
    df.to_csv(file_name, index=False, encoding='utf-8')

    # 打印保存结果文件的绝对路径
    file_path = os.path.abspath(file_name)
    print(f"\nThe response file has been saved to:\n{file_path}\n")

if __name__ == "__main__":
    # 创建ArgumentParser对象以解析命令行参数
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument('--data_path', type=str, default=r"E:\我的论文和代码\ReNeLLM\data\advbench\harmful_behaviors.csv", help='data path to your renellm jailbroken json file')
    parser.add_argument('--baseline', type=str, default="renellm", choices=['renellm', 'gcg', 'autodan', 'pair'], help='which baseline to use')
    parser.add_argument('--test_model', type=str, default="deepseek-R1", 
                        choices=["deepseek-R1", "gpt-4", \
                                 "anthropic.claude-instant-v1", "anthropic.claude-v2", "mistral"], help='which model to test')
    parser.add_argument('--temperature', type=float, default=0, help='model temperature')
    parser.add_argument("--max_tokens", type=int, default=3584)
    parser.add_argument('--round_sleep', type=int, default=5, help='sleep time between every round')
    parser.add_argument('--fail_sleep', type=int, default=5, help='sleep time for fail response')
    parser.add_argument('--retry_times', type=int, default=1000, help='retry times when exception occurs')
    parser.add_argument('--save_suffix', type=str, default="normal")
    parser.add_argument("--gpt_api_key", type=str, default=None)
    parser.add_argument("--claude_api_key", type=str, default=None)
    parser.add_argument("--gpt_base_url", type=str, default=None)
    parser.add_argument("--claude_base_url", type=str, default=None)

    # 解析命令行参数
    args = parser.parse_args()

    # 调用main函数并传入解析后的参数
    # 添加 DeepSeek 相关参数
    parser.add_argument("--deepseek_api_key", type=str, default="sk-wrjmrfodxdmizbtjfagnfmetjfsqqcbcjjiwmyflnijgmjwm")
    parser.add_argument("--deepseek_base_url", type=str, default="https://api.siliconflow.cn/v1")
    
    args = parser.parse_args()
    main(args)



