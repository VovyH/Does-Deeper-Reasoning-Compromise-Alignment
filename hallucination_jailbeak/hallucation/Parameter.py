"""
配置模块 - 管理FreeKnowledge AI的全局配置项
"""

import os
from dataclasses import dataclass

# 定义项目根目录（所有路径基于此目录动态生成）
BASE_DIR = r"D:\个人信息\Code\2025ACL_CHJ\2025ACL_CHJ\CHJ\hallucination_jailbeak"
@dataclass
class Parameter:
    chat_model_type: str = "Qwen/Qwen3-14B"
    model_path: str = "claude"
    judge_model: str = "Pro/deepseek-ai/DeepSeek-V3"
    feihua_key:str = "sk-qtqstxbfqbekatkcqenngmnbvgbuusibuikezybrqznittfl"
    judge_model_key: str = "sk-cxgbkcslxrnbujqrmkklgocmofbqzkguqdllxhbcyitndwbf"
    motivation_key: str = "sk-enzwnozvwlakthixwbsmstlrpayefwfbraigfartdcxbciyd"
    jailbreak_key: str = "sk-nvifhtqcjbwnlzldxcbggpykvywajrghylundqsaqlvufdec"
    model_key_2: str = "sk-enzwnozvwlakthixwbsmstlrpayefwfbraigfartdcxbciyd"
    other_key_1: str = "sk-ivaemxaqkivxbppefrbkzlcqhaahaldwgzttyxfdeuzjhume"
    other_key_2: str = "sk-jecqzrvjjccvpswqnnsvpgurnsnijsmgeufyxkmytcbnqghk"
    other_key_3: str = "sk-oyrjoybqtmxijzbjvoegquxeleiyzcbebznphlyeoxdtfmfb"
    other_key_4: str = "sk-eeitdlhtzcczxujrdinppsnpoxcdbabwhhcvjtsgxykjmsnn"
    other_key_5: str = "sk-xynaxnaqndimkbshtetngoqxadspdolkueefvumhyhtedsrm"
    model_base_url: str = "https://api.siliconflow.cn/v1/chat/completions"
    base_url: str = "https://api.siliconflow.cn/v1/"
    max_tokens: int = 8192
    temperature: float = 0.6
    enable_thinking: bool = False
    thinking_budget: int = 1
    mode: str = "no_thinking"
    # 动态路径配置（基于BASE_DIR拼接）
    backend_dir: str = os.path.join(BASE_DIR, "result/function/")
    AIME_PATH: str = os.path.join(BASE_DIR, "data/aime_test.csv")
    Asker_PATH: str = os.path.join(BASE_DIR, "data/asker.csv")
    Asker_feihua_PATH: str = os.path.join(BASE_DIR, "data/asker_feihua.csv")
    Asker_PAP_PATH: str = os.path.join(BASE_DIR, "data/asker_pap.csv")
    AIME_feihua_PATH: str = os.path.join(BASE_DIR, "data/aime_feihua.csv")
    AIME_PAP_PATH: str = os.path.join(BASE_DIR, "data/aime_pap.csv")
    METHOD: str = "no"
    run_count: int = 4

    Jailbreak_backend_dir: str = os.path.join(BASE_DIR, "result/jailbreak_result/")
    Jailbreak_DATA: str = os.path.join(BASE_DIR, "data/harmful_behaviors.csv")
    Jailbreak_Method: str = "PAP"
    Jailbreak_key:str = "sk-eiggjmqmyvclhkbnybnfdisghuajyydsvtdyoqffekjmtxsk"

    jailbreak_key_gpt:str = "sk-5nDBIZh3dHWhiv9d9yU6cGobp8zYJ0fqy6atuMo3X1fOn64y"
    attacker_model1:str = "llama-3.1-8b"
    attacker_model2:str = "deepseek-v3-250324"
    attacker_model3:str = "gpt-5-2025-08-07"
    attacker_model3:str = "llama2-7B"
Parameter_args = Parameter()
