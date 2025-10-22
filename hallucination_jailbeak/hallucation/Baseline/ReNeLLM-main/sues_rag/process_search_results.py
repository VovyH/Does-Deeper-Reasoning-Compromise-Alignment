import re
from typing import Optional

def extract_chinese(text: str) -> Optional[str]:
    """
    提取字符串中的所有中文字符，保留换行符。
    
    Args:
        text: 输入字符串，可能包含中文、英文、数字、符号等，包含换行符
        
    Returns:
        提取的中文字符串（保留换行符），如果无中文则返回空字符串，如果输入无效则返回 None
    """
    try:
        if not isinstance(text, str):
            raise ValueError("输入必须是字符串")
        
        # 使用正则表达式匹配中文字符（Unicode 范围 \u4e00-\u9fff）和换行符
        pattern = r'[\u4e00-\u9fff\n]'
        matches = re.findall(pattern, text)
        
        # 将匹配结果拼接为字符串
        result = ''.join(matches)
        
        # 如果结果为空，返回空字符串
        return result if result else ""
    
    except Exception as e:
        print(f"提取中文时出错: {str(e)}")
        return None

def main():
    # 测试用例
    test_cases = [
        "Hello 世界！\nThis is a 测试 string.",
        "没有中文",
        "中文\n换行\n测试",
        "123 中文 abc \n 456 汉字 def",
        "",  # 空字符串
        None,  # 无效输入
        "纯中文：你好，世界！\n再见",
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}:")
        print(f"输入: {repr(test)}")
        result = extract_chinese(test)
        print(f"输出: {repr(result)}")
        print("-" * 50)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    # 输入输出文件路径
    input_json = r"e:\我的论文和代码\Chemotherapy\ReNeLLM-main\sues_rag\duckduckgo_search_results.json"
    output_json = r"e:\我的论文和代码\Chemotherapy\ReNeLLM-main\sues_rag\processed_search_results.json"
    
    # 处理并保存结果
    results = process_search_results(input_json, output_json)
    print(f"处理完成，共处理 {len(results)} 条结果，已保存到 {output_json}")