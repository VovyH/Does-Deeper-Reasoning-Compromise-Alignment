"""
配置模块 - 管理FreeKnowledge AI的全局配置项
"""

import os
from dataclasses import dataclass

# 定义项目根目录（所有路径基于此目录动态生成）
BASE_DIR = r"E:/我的论文和代码/LLM_Study/hallucination_jailbeak"

@dataclass
class Parameter:
    chat_model_type: str = "Qwen/Qwen3-14B"
    model_path: str = "Qwen3-14B"
    judge_model: str = "Pro/deepseek-ai/DeepSeek-V3"
    judge_model_key: str = "sk-zdgsvrykdfkoqymhzqnttceswdvmthjszzcdrreihanobfxr"
    model_key: str = "sk-dpluqdhaptliuhmtnlvvkwrghbawhqczfqlremhpojyjbcxh"
    model_key_2: str = "sk-wrjmrfodxdmizbtjfagnfmetjfsqqcbcjjiwmyflnijgmjwm"
    other_key_1: str = "sk-ivaemxaqkivxbppefrbkzlcqhaahaldwgzttyxfdeuzjhume"
    other_key_2: str = "sk-jecqzrvjjccvpswqnnsvpgurnsnijsmgeufyxkmytcbnqghk"
    other_key_3: str = "sk-oyrjoybqtmxijzbjvoegquxeleiyzcbebznphlyeoxdtfmfb"
    other_key_4: str = "sk-eeitdlhtzcczxujrdinppsnpoxcdbabwhhcvjtsgxykjmsnn"
    other_key_5: str = "sk-xynaxnaqndimkbshtetngoqxadspdolkueefvumhyhtedsrm"
    model_base_url: str = "https://api.siliconflow.cn/v1/chat/completions"
    max_tokens: int = 8192
    temperature: float = 0.6
    enable_thinking: bool = True
    thinking_budget: int = 4096
    API_KEY: str = model_key
    mode: str = "thinking"
    # 动态路径配置（基于BASE_DIR拼接）
    backend_dir: str = os.path.join(BASE_DIR, "result/function/")
    AIME_PATH: str = os.path.join(BASE_DIR, "data/aime_test.csv")
    AIME_PATH_METHOD: str = os.path.join(BASE_DIR, "data/aime_feihua.csv")
    AIME_PAP_PATH: str = os.path.join(BASE_DIR, "data/aime_pap.csv")
    METHOD: str = "false"
    run_count: int = 8
Parameter_args = Parameter()