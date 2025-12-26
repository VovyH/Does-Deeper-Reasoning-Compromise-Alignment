import pandas as pd
import os

def add_ground_truth_column():
    # 1. 定义文件路径
    # 源数据文件路径 (提供 output 列)
    source_csv_path = r"D:\个人信息\Code\2025ACL_CHJ\2025ACL_CHJ\CHJ\hallucination_jailbeak\data\aime_test.csv"
    
    # 目标文件夹路径 (需要插入 ground_truth 列)
    target_dir = r"D:\个人信息\Code\2025ACL_CHJ\2025ACL_CHJ\CHJ\hallucination_jailbeak\result\aime\feihua"

    # 2. 读取源数据
    if not os.path.exists(source_csv_path):
        print(f"❌ 错误：源文件不存在 -> {source_csv_path}")
        return

    print(f"📖 正在读取源文件: {os.path.basename(source_csv_path)}...")
    df_source = pd.read_csv(source_csv_path)

    # 提取 ground_truth 数据
    ground_truth_series = df_source['output']
    source_len = len(df_source)

    # 3. 遍历目标文件夹下的所有 CSV 文件
    if not os.path.exists(target_dir):
        print(f"❌ 错误：目标目录不存在 -> {target_dir}")
        return

    files = [f for f in os.listdir(target_dir) if f.endswith('.csv')]
    
    if not files:
        print(f"⚠️ 警告：在 {target_dir} 中没有找到 CSV 文件。")
        return

    print(f"🔍 找到 {len(files)} 个目标 CSV 文件，准备处理...")

    for filename in files:
        file_path = os.path.join(target_dir, filename)
        
        try:
            # 读取目标文件
            df_target = pd.read_csv(file_path)
            target_len = len(df_target)

            # --- 安全检查：行数是否匹配 ---
            if target_len != source_len:
                print(f"⚠️ 跳过文件 {filename}：行数不匹配 (源文件: {source_len} vs 目标文件: {target_len})")
                continue
            
            # --- 核心操作：赋值 ---
            # 直接按顺序赋值 (假设行顺序是一一对应的)
            df_target['ground_truth'] = ground_truth_series

            # 保存回原路径 (覆盖原文件)
            df_target.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"✅ 成功修改: {filename}")

        except Exception as e:
            print(f"❌ 处理文件 {filename} 时出错: {e}")

    print("\n🎉 所有处理完成！")

if __name__ == "__main__":
    add_ground_truth_column()