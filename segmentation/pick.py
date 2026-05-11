"""
用于对多个text中的F1进行排序。
"""

import os

# 设置目标文件夹路径
# folder_path = r'./experiments/CRAG/buchang8/test'
base_path = r'./experiments/CRAG_attunet/test'
folder_path = os.path.join(base_path, 'test')

num = 0.60

# 存储结果
results = []
total_files = 0
valid_files = 0

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 只处理txt文件
    if filename.endswith('.txt'):
        total_files += 1
        file_path = os.path.join(folder_path, filename)
        try:
            # 打开文件并读取内容
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                # 确保文件至少有两行内容
                if len(lines) >= 2:
                    # 获取第二行并分割数据（假设使用制表符分隔）
                    second_line = lines[1].strip()
                    values = second_line.split('\t')
                    # 确保有足够的数据列
                    if len(values) >= 4:
                        # 获取倒数第四个数并转换为浮点数
                        target_value = float(values[-4])
                        # 检查是否满足条件
                        if target_value >= num:
                            # 存储文件名、目标值和后四个数
                            last_four_values = values[-4:]  # 获取后四个数
                            results.append((filename, target_value, last_four_values))
                            valid_files += 1
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")

# 按值的大小降序排序
results.sort(key=lambda x: x[1], reverse=True)

# 设置输出文件路径
output_file = os.path.join(base_path, 'test_best_model_info.txt')

# 将结果写入文件
with open(output_file, 'w', encoding='utf-8') as f:
    for filename, target_value, last_four_values in results:
        # 将后四个数用空格连接并输出
        last_four_str = ' '.join(last_four_values)
        f.write(f"{filename}: {last_four_str}\n")
    
    # 写入统计信息
    f.write(f"\n设置的阈值为{num}\n")
    f.write(f"共{total_files}个，符合要求有{valid_files}个\n")

# 输出统计信息到控制台
print(f"结果已保存到: {output_file}")
print(f"设置的阈值为{num}")
print(f"共{total_files}个，符合要求有{valid_files}个")