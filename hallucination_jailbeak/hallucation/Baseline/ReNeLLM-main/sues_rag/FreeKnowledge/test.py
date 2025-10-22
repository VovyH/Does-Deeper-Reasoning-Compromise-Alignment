from FreeKnowledge_AI import knowledge_center

center = knowledge_center.Center()
question = "2025年EMNLP会议的主题是什么?"
flag = True 
mode = "DUCKDUCKGO"

results = center.get_response(question, flag, mode, model="internlm/internlm2_5-7b-chat", 
                              base_url="https://api.siliconflow.cn/v1/chat/completions", key = "xxx", max_web_results = 2)
print(results) 