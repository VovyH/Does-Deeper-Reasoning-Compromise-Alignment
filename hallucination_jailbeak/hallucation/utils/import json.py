import json
import csv
import ast
import os
import sys
from pathlib import Path

def validate_file_path(file_path: str, expected_suffixes: list) -> Path:
    """
    验证文件路径合法性和后缀名
    
    Args:
        file_path: 文件路径字符串
        expected_suffixes: 期望的文件后缀列表（如 ['.csv', '.json']）
    
    Returns:
        验证通过的Path对象
    
    Raises:
        ValueError: 路径不合法或后缀不符
        FileNotFoundError: 文件不存在（仅针对输入文件）
    """
    # 转换为Path对象
    try:
        file_path_obj = Path(file_path).resolve()
    except Exception as e:
        raise ValueError(f"文件路径格式非法: {file_path}, 错误: {str(e)}")
    
    # 检查后缀
    suffix = file_path_obj.suffix.lower()
    if suffix not in expected_suffixes:
        raise ValueError(
            f"文件后缀不合法！期望: {expected_suffixes}, 实际: {suffix}, 文件: {file_path}"
        )
    
    return file_path_obj

def validate_input_file_exists(file_path: Path) -> None:
    """验证输入文件存在"""
    if not file_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {file_path}")
    
    # 检查文件是否为空
    if file_path.stat().st_size == 0:
        raise ValueError(f"输入文件为空: {file_path}")

def validate_csv_structure(file_path: Path, required_columns: list = None) -> None:
    """
    验证CSV文件结构合法性
    
    Args:
        file_path: CSV文件路径
        required_columns: 必需的列名列表（可选）
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 检查是否是有效的CSV
            dialect = csv.Sniffer().sniff(f.read(1024))
            f.seek(0)
            
            # 检查列名（如果指定了必需列）
            if required_columns:
                reader = csv.DictReader(f, dialect=dialect)
                csv_columns = reader.fieldnames or []
                missing_cols = [col for col in required_columns if col not in csv_columns]
                if missing_cols:
                    raise ValueError(f"CSV文件缺少必需列: {missing_cols}, 现有列: {csv_columns}")
    except csv.Error as e:
        raise ValueError(f"CSV文件格式非法: {file_path}, 错误: {str(e)}")
    except Exception as e:
        raise ValueError(f"验证CSV文件失败: {file_path}, 错误: {str(e)}")

def validate_json_structure(file_path: Path) -> tuple:
    """
    验证JSON文件结构并加载数据（兼容字典/列表两种根结构）
    
    Args:
        file_path: JSON文件路径
    
    Returns:
        tuple: (数据类型标识, 加载的JSON数据)
               数据类型标识: 'dict' 或 'list'
    
    Raises:
        ValueError: JSON格式非法或数据为空
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 加载JSON数据
            data = json.load(f)
            
            # 检查数据是否为空
            if not data:
                raise ValueError("JSON数据为空")
            
            # 识别根结构类型
            if isinstance(data, dict):
                data_type = 'dict'
                # 如果是字典，检查是否包含malicious_details字段（可选）
                if 'malicious_details' in data:
                    malicious_details = data.get('malicious_details', [])
                    if not isinstance(malicious_details, list):
                        raise ValueError(f"malicious_details必须是列表，实际类型: {type(malicious_details).__name__}")
                    if len(malicious_details) == 0:
                        raise ValueError("malicious_details列表为空")
                    # 验证列表元素是字典
                    if malicious_details and not isinstance(malicious_details[0], dict):
                        raise ValueError(f"malicious_details元素必须是字典，实际类型: {type(malicious_details[0]).__name__}")
            elif isinstance(data, list):
                data_type = 'list'
                # 如果是列表，验证元素是字典
                if not isinstance(data[0], dict):
                    raise ValueError(f"JSON列表元素必须是字典，实际类型: {type(data[0]).__name__}")
            else:
                raise ValueError(f"JSON文件根结构必须是字典或列表，实际类型: {type(data).__name__}")
            
            return (data_type, data)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON格式非法: {file_path}, 错误位置: {e.pos}, 错误: {e.msg}")
    except Exception as e:
        raise ValueError(f"读取JSON文件失败: {file_path}, 错误: {str(e)}")

# 函数1：处理CSV文件中某列的JSON数据，提取指定字段到新CSV
def column_json_to_csv(input_file: str, output_file: str, column: str):
    """
    处理CSV文件中某列的JSON数据，提取指定字段到新CSV
    
    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        column: 要解析的包含JSON数据的列名
    
    Raises:
        各类验证错误和处理错误
    """
    # ===================== 前置验证 =====================
    try:
        # 验证输入输出文件路径和后缀
        input_path = validate_file_path(input_file, ['.csv'])
        output_path = validate_file_path(output_file, ['.csv'])
        
        # 验证输入文件存在且合法
        validate_input_file_exists(input_path)
        validate_csv_structure(input_path, [column, 'prompt'])  # 确保必需列存在
        
        # 检查输出目录是否存在，不存在则创建
        output_dir = output_path.parent
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"开始处理CSV文件: {input_path}")
        print(f"输出文件路径: {output_path}")
        
    except Exception as e:
        print(f"前置验证失败，终止处理！错误: {str(e)}")
        return

    # ===================== 数据处理 =====================
    processed_count = 0
    skipped_count = 0
    
    try:
        with open(input_path, mode='r', encoding='utf-8') as infile, \
             open(output_path, mode='w', newline='', encoding='utf-8') as outfile:

            reader = csv.DictReader(infile)
            fieldnames = ['prompt', 'content', 'reasoning_tokens', 'completion_tokens']
            writer = csv.writer(outfile)
            writer.writerow(fieldnames)
            
            for idx, data_entity in enumerate(reader, 1):
                try:
                    # 验证行数据是字典
                    if not isinstance(data_entity, dict):
                        skipped_count += 1
                        print(f"第 {idx} 行: 数据不是字典格式，跳过")
                        continue
                    
                    # 获取基础字段并验证
                    prompt = data_entity.get('prompt', '')
                    column_str = data_entity.get(column, '')
                    
                    # 空值检查
                    if not column_str or column_str.strip() in ['{}', '[]', 'null']:
                        skipped_count += 1
                        writer.writerow([prompt, '', 0, 0])
                        continue
                    
                    # 解析JSON数据
                    try:
                        column_json = json.loads(column_str.strip())
                    except json.JSONDecodeError:
                        try:
                            column_json = ast.literal_eval(column_str.strip())
                        except:
                            skipped_count += 1
                            print(f"第 {idx} 行: JSON/字面量解析失败，跳过")
                            writer.writerow([prompt, '', 0, 0])
                            continue
                    
                    # 验证解析结果是字典
                    if not isinstance(column_json, dict):
                        skipped_count += 1
                        print(f"第 {idx} 行: 解析结果不是字典，跳过")
                        writer.writerow([prompt, '', 0, 0])
                        continue
                    
                    # 安全提取嵌套字段
                    choices = column_json.get("choices", [])
                    content = ""
                    completion_tokens = 0
                    reasoning_tokens = 0
                    
                    if isinstance(choices, list) and len(choices) > 0:
                        first_choice = choices[0]
                        if isinstance(first_choice, dict):
                            message = first_choice.get("message", {})
                            if isinstance(message, dict):
                                content = message.get("content", "")
                    
                    usage = column_json.get("usage", {})
                    if isinstance(usage, dict):
                        completion_tokens = usage.get("completion_tokens", 0)
                        completion_details = usage.get("completion_tokens_details", {})
                        if isinstance(completion_details, dict):
                            reasoning_tokens = completion_details.get("reasoning_tokens", 0)
                    
                    # 写入有效数据
                    writer.writerow([prompt, content, reasoning_tokens, completion_tokens])
                    processed_count += 1
                    
                except Exception as e:
                    skipped_count += 1
                    print(f"第 {idx} 行: 处理异常 - {str(e)}，跳过")
                    writer.writerow(['', '', 0, 0])
                    continue
        
        # 输出处理统计
        total_count = processed_count + skipped_count
        print(f"\n处理完成！")
        print(f"总行数: {total_count}")
        print(f"成功处理: {processed_count}")
        print(f"跳过行数: {skipped_count}")
        print(f"输出文件: {output_path}")
        
    except Exception as e:
        print(f"\n处理过程中发生严重错误: {str(e)}")
        # 清理空的输出文件
        if output_path.exists() and output_path.stat().st_size == 0:
            output_path.unlink()
        return

# 函数2：处理JSON文件（兼容字典/列表根结构），提取字段到CSV
def json_to_csv(json_file_path: str, csv_file_path: str):
    """
    处理JSON文件（兼容字典/列表两种根结构），提取字段到CSV
    
    Args:
        json_file_path: 输入JSON文件路径
        csv_file_path: 输出CSV文件路径
    
    Raises:
        各类验证错误和处理错误
    """
    # ===================== 前置验证 =====================
    try:
        # 验证文件路径和后缀
        json_path = validate_file_path(json_file_path, ['.json'])
        csv_path = validate_file_path(csv_file_path, ['.csv'])
        
        # 验证输入JSON文件存在且合法
        validate_input_file_exists(json_path)
        
        # 加载并验证JSON结构（兼容字典/列表）
        data_type, json_data = validate_json_structure(json_path)
        
        # 确定核心数据列表
        if data_type == 'dict':
            # 字典结构：从malicious_details提取数据
            core_data = json_data.get('malicious_details', [])
            # 获取元数据
            meta_data = {
                'file_path': json_data.get('file_path', '未知'),
                'total_responses': json_data.get('total_responses', ''),
                'malicious_count': json_data.get('malicious_count', ''),
                'malicious_percentage': json_data.get('malicious_percentage', '')
            }
        else:
            # 列表结构：直接使用根列表作为核心数据
            core_data = json_data
            # 列表结构无元数据
            meta_data = {
                'file_path': '未知',
                'total_responses': '',
                'malicious_count': '',
                'malicious_percentage': ''
            }
        
        # 检查核心数据是否为空
        if len(core_data) == 0:
            raise ValueError("JSON核心数据列表为空")
        
        # 检查输出目录
        output_dir = csv_path.parent
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"开始处理JSON文件: {json_path}")
        print(f"JSON根结构类型: {data_type}")
        print(f"输出CSV文件: {csv_path}")
        print(f"核心数据条数: {len(core_data)}")
        print(f"JSON文件元信息:")
        print(f"  - 文件路径: {meta_data['file_path']}")
        print(f"  - 总响应数: {meta_data['total_responses']}")
        print(f"  - 恶意响应数: {meta_data['malicious_count']}")
        print(f"  - 恶意响应占比: {meta_data['malicious_percentage']}%")
        
    except Exception as e:
        print(f"前置验证失败，终止处理！错误: {str(e)}")
        return

    # ===================== 数据处理 =====================
    processed_count = 0
    skipped_count = 0
    
    try:
        with open(csv_path, 'w', encoding='utf-8', newline='') as csv_file:
            # 定义CSV列名
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                'id', 'prompt', 'is_harmful', 'content', 
                'total_responses', 'malicious_count', 'malicious_percentage'
            ])
            
            # 遍历核心数据列表
            for idx, data_entity in enumerate(core_data, 1):
                try:
                    # 验证数据实体是字典
                    if not isinstance(data_entity, dict):
                        skipped_count += 1
                        print(f"第 {idx} 条: 数据不是字典格式，跳过")
                        continue
                    
                    # 提取核心字段（兼容不同结构的字段名）
                    entity_id = data_entity.get("id", idx)  # 没有id则用索引
                    prompt = data_entity.get("goal", data_entity.get("prompt", ""))  # 兼容goal/prompt字段
                    is_harmful = data_entity.get("is_harmful", "")
                    content = data_entity.get("output", data_entity.get("content", ""))  # 兼容output/content字段
                    
                    # 字段类型处理
                    if not isinstance(content, str):
                        content = str(content)
                    
                    # 写入数据（包含元数据）
                    csv_writer.writerow([
                        entity_id, prompt, is_harmful, content,
                        meta_data['total_responses'], meta_data['malicious_count'], meta_data['malicious_percentage']
                    ])
                    processed_count += 1
                    
                except Exception as e:
                    skipped_count += 1
                    print(f"第 {idx} 条: 处理异常 - {str(e)}，跳过")
                    continue
        
        # 输出处理统计
        print(f"\n处理完成！")
        print(f"总数据条数: {len(core_data)}")
        print(f"成功处理: {processed_count}")
        print(f"跳过条数: {skipped_count}")
        print(f"输出文件: {csv_path}")
        
    except Exception as e:
        print(f"\n处理过程中发生严重错误: {str(e)}")
        # 清理空文件
        if csv_path.exists() and csv_path.stat().st_size == 0:
            csv_path.unlink()
        return

# ------------------- 主执行逻辑 -------------------
if __name__ == "__main__":
    # 设置异常处理
    def handle_exception(exc_type, exc_value, exc_traceback):
        """全局异常处理"""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        print(f"\n程序异常终止！")
        print(f"异常类型: {exc_type.__name__}")
        print(f"异常信息: {exc_value}")
    
    sys.excepthook = handle_exception
    
    # 示例调用
    try:
        # 1. 处理JSON文件→CSV（兼容字典/列表结构）
        json_input = r"E:\my_code\FlipAttack\without_cot_result\result-chain-flipattack-deepseekv3-FCW-deepseek-v3-advbench.json"
        csv_output = r"D:\个人信息\Code\2025ACL_CHJ\2025ACL_CHJ\CHJ\hallucination_jailbeak\result\jailbreak_result\result-chain-flipattack-deepseekv3.csv"
        
        json_to_csv(json_input, csv_output)
        
        # 2. 若需处理CSV中某列的JSON数据（示例用法）
        # csv_input = r"xxx.csv"
        # csv_col_output = r"xxx_col_output.csv"
        # column_json_to_csv(csv_input, csv_col_output, column="content")
        
    except Exception as e:
        print(f"\n执行失败: {str(e)}")
        sys.exit(1)
    
    print("\n所有任务执行完成！")