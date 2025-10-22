from bs4 import BeautifulSoup
import requests
import io
import pdfplumber

def remove_tags(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()
    text = ' '.join(text.split())
    return text

def extract_pdf_content(pdf_url):
    """直接从URL读取PDF内容"""
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        
        # 使用内存中的PDF文件
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            text_content = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_content += text + "\n"
        return text_content.strip()
    except Exception as e:
        print(f"读取PDF内容时出错: {e}")
        return None

def extract_specific_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 尝试获取文章标题
    title = soup.find('h1', class_='arti_title')
    title_text = title.get_text(strip=True) if title else "无标题"
    
    # 尝试获取发布日期
    date = soup.find('span', class_='arti_update')
    date_text = date.get_text(strip=True) if date else "无日期"
    
    # 检查是否有PDF内容
    pdf_div = soup.find('div', class_='wp_articlecontent')
    if pdf_div and pdf_div.find('div', attrs={'pdfsrc': True}):
        pdf_src = pdf_div.find('div', attrs={'pdfsrc': True})['pdfsrc']
        
        # 确保URL完整
        if not pdf_src.startswith('http'):
            pdf_src = 'https://ge.sues.edu.cn' + pdf_src
        
        # 直接读取PDF内容
        pdf_text = extract_pdf_content(pdf_src)
        
        if pdf_text:
            return {
                'title': title_text,
                'date': date_text,
                'content_type': 'pdf',
                'pdf_url': pdf_src,
                'content': pdf_text
            }
        else:
            return {
                'title': title_text,
                'date': date_text,
                'content_type': 'pdf',
                'pdf_url': pdf_src,
                'message': '无法读取PDF内容'
            }
    
    # 如果没有PDF，尝试获取普通文本内容
    content_tag = soup.find('div', class_='wp_articlecontent')
    if content_tag:
        content_text = content_tag.get_text()
        content_text = ' '.join(content_text.split())
        return {
            'title': title_text,
            'date': date_text,
            'content': content_text
        }
    
    return {
        'title': title_text,
        'date': date_text,
        'message': '未找到正文内容'
    }

html_content = """
       
"""

# 提取所有文本
all_text = remove_tags(html_content)
print("所有文本内容：")
print(all_text)

# 提取特定内容
specific_content = extract_specific_content(html_content)
print("\n特定内容：")
print(specific_content)