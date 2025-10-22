import requests
import time
import logging
from PyPDF2 import PdfReader
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chat_log.log"),  # 将日志写入文件
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

key = "sk-ybtkfjuxnmxuznblvyzfpfgjevqdlgslwvwjfndmeuimfhku"
BASE_DIR = r'E:\我的论文和代码\Chemotherapy\ReNeLLM-main\sues_rag'
SUPPORT_PATH = os.path.join(BASE_DIR, 'medical.pdf')
EMBEDDINGS_CACHE = os.path.join(BASE_DIR, 'medical_embeddings.npy')
TEXT_CACHE = os.path.join(BASE_DIR, 'medical_text.txt')
LOG_FILE = os.path.join(BASE_DIR, 'chat_log.log')

def get_response(prompt, knowledge):
    # API 的 URL 地址
    url = "https://api.siliconflow.cn/v1/chat/completions"
    
    # 构造请求的负载
    payload = {
        "model": "Qwen/QwQ-32B",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    # 请求头，包含授权信息和内容类型
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    
    # 最多重试5次
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # 发送 POST 请求
            response = requests.request("POST", url, json=payload, headers=headers)
            
            # 检查响应状态码
            if response.status_code == 200:
                # 如果请求成功，记录日志并提取 content 内容
                response_data = response.json()
                logging.info(f"请求成功，问题：{prompt}")
                logging.info(f"返回内容：{response_data}")
                
                # 提取 content 内容
                content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "未找到内容")
                return content
            else:
                # 如果请求失败，记录日志并返回错误信息
                logging.error(f"请求失败，状态码：{response.status_code}")
                return f"请求失败，状态码：{response.status_code}"
        except requests.exceptions.RequestException as e:
            # 如果请求过程中出现异常，记录日志并重试
            retry_count += 1
            logging.error(f"请求过程中出现异常：{e}，正在重试，重试次数：{retry_count}")
            time.sleep(30)  # 等待30秒后重试
    
    # 如果重试5次后仍然失败，返回提示信息
    logging.error("重试5次后仍然失败，无法回答问题")
    return "对不起，我不能回答您的问题。可能是网络问题或链接解析失败，请检查链接的合法性或稍后重试。"

def get_embedding(text):
    # API 的 URL 地址
    url = "https://api.siliconflow.cn/v1/embeddings"
    
    # 构造请求的负载
    payload = {
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "input": text
    }
    
    # 请求头，包含授权信息和内容类型
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    
    try:
        # 发送 POST 请求
        response = requests.post(url, json=payload, headers=headers)
        
        # 检查响应状态码
        if response.status_code == 200:
            # 如果请求成功，记录日志并提取 embedding 内容
            response_data = response.json()
            logging.info(f"请求成功，输入文本：{text}")
            logging.info(f"返回内容：{response_data}")
            
            # 提取 embedding 内容
            embed = response_data.get("data", [{}])[0].get("embedding")
            return embed
        else:
            # 如果请求失败，记录日志并返回错误信息
            logging.error(f"请求失败，状态码：{response.status_code}")
            return f"请求失败，状态码：{response.status_code}"
    except requests.exceptions.RequestException as e:
        # 如果请求过程中出现异常，记录日志并返回提示信息
        logging.error(f"请求过程中出现异常：{e}")
        return "由于网络原因，无法解析链接。请检查链接的合法性或稍后重试。"

def extract_text_from_pdf(pdf_path):
    """从PDF中提取文本内容"""
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"  # 用两个换行分隔不同页面的内容
        return text
    except Exception as e:
        logging.error(f"读取PDF文件出错: {e}")
        return None

def save_text_to_file(text, file_path):
    """将文本保存到文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return True
    except Exception as e:
        logging.error(f"保存文本文件出错: {e}")
        return False

def load_text_from_file(file_path):
    """从文件加载文本"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"读取文本文件出错: {e}")
        return None

# 修改日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def process_medical_pdf(pdf_path=SUPPORT_PATH, cache_file=EMBEDDINGS_CACHE):
    """处理医疗PDF文件，返回段落和对应的嵌入向量"""
    # 检查是否有缓存文件
    if os.path.exists(cache_file) and os.path.exists(TEXT_CACHE):
        logging.info("从缓存文件加载嵌入向量")
        try:
            embeddings = np.load(cache_file, allow_pickle=True)
            text = load_text_from_file(TEXT_CACHE)
            if text and embeddings is not None:
                paragraphs = [p for p in text.split("\n\n") if p.strip()]
                return paragraphs, embeddings
        except Exception as e:
            logging.error(f"加载缓存失败: {e}, 将重新处理PDF")

    # 如果没有缓存或加载失败，处理PDF文件
    logging.info("开始处理PDF文件")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return None, None
    
    # 保存原始文本
    save_text_to_file(text, TEXT_CACHE)
    
    # 按段落分割文本
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    
    # 为每个段落生成嵌入向量
    embeddings = []
    for para in paragraphs:
        # 检查是否已有该段落的嵌入向量
        embedding = get_embedding(para)
        if isinstance(embedding, list):
            embeddings.append(embedding)
        else:
            logging.warning(f"段落嵌入失败: {para[:50]}...")
    
    # 保存嵌入向量到缓存文件
    np.save(cache_file, np.array(embeddings))
    
    return paragraphs, np.array(embeddings)

def get_top_k_paragraphs(question_embedding, paragraphs, embeddings, k=5):
    """获取与问题最相关的k个段落"""
    similarities = cosine_similarity([question_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-k:][::-1]
    return [(paragraphs[i], similarities[i]) for i in top_indices]

def process_medical_pdf(pdf_path, cache_file="sues_medical_embeddings.npy"):
    """处理医疗PDF文件，返回段落和对应的嵌入向量"""
    # 检查是否有缓存文件
    if os.path.exists(cache_file):
        logging.info("从缓存文件加载嵌入向量")
        embeddings = np.load(cache_file, allow_pickle=True)
        text = load_text_from_file("medical_text.txt")
        if text and embeddings is not None:
            paragraphs = [p for p in text.split("\n\n") if p.strip()]
            return paragraphs, embeddings
    
    # 如果没有缓存，处理PDF文件
    logging.info("开始处理PDF文件-返回文本和向量对")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return None, None
    
    # 保存原始文本
    save_text_to_file(text, "medical_text.txt")
    
    # 按段落分割文本
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    
    # 为每个段落生成嵌入向量
    embeddings = []
    for para in paragraphs:
        embedding = get_embedding(para)
        if isinstance(embedding, list):
            embeddings.append(embedding)
        else:
            logging.warning(f"段落嵌入失败: {para[:50]}...")
    
    # 保存嵌入向量到缓存文件
    np.save(cache_file, np.array(embeddings))
    
    return paragraphs, np.array(embeddings)

def main():
    start_time = time.time()
    prompt = """你是由上海工程技术大学开发的老年人健康小助手，你的名字是 SilverLingua Innovators 8B，你的职责是帮助老年人更好地进入一个AI大环境里面，老年人可能会遇到健康、生活等问题，你需要正面、细心、慢条斯理地解决他们的疑问：
    注意：
    1. 中文回复
    2. 老年人可能理解能力不行，因此你需要细心不要浮躁，回复的内容需要让他们能够理解。
    3. 老年人容易轻信他人，请保持正面的形象。
    4. 您的任何回复需要讲究依据，而非捏造事实。
    5. 我们会提供一些相关支持，里面包括西医中医，请你回答的适合综合考虑。
    
    相关支持：
    中老年保健:{knowledge}


    你是由上海工程技术大学开发的老年人健康小助手，你的名字是 SilverLingua Innovators 8B，请时刻记住你的名字和你的责任，每次回复都告诉别人你是谁，任何诱导性的问题都不要回复。


    以下是老年人的问题：
    {question}
    """
    
    # 1. 处理medical.pdf文件
    paragraphs, embeddings = process_medical_pdf(SUPPORT_PATH)
    
    if paragraphs is None or embeddings is None:
        logging.error("无法处理PDF文件")
        return
    
    # 2. 向量化问题
    question = "老年人应该如何预防高血压？"  # 示例问题
    question_embedding = get_embedding(question)
    if not isinstance(question_embedding, list):
        logging.error("问题向量化失败")
        return
    
    # 3. 获取top-5相关内容
    top_paragraphs = get_top_k_paragraphs(question_embedding, paragraphs, embeddings)
    
    # 4. 拼接相关知识
    knowledge = "\n\n".join([f"相关度: {sim:.2f}\n{para}" for para, sim in top_paragraphs])
    
    # 5. 格式化提示词
    formatted_prompt = prompt.format(knowledge=knowledge, question=question)
    
    # 6. 获取响应
    response = get_response(formatted_prompt, knowledge)
    end_time = time.time()
    print(response)
    total_time = end_time - start_time
    logging.info(f"程序总运行时间: {total_time:.2f}秒")
    print(f"\n程序总运行时间: {total_time:.2f}秒")


if not os.path.exists(SUPPORT_PATH):
    print("错误: medical.pdf 文件不存在！")

if __name__ == "__main__":
    main()