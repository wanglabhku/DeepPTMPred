import pandas as pd
import os

# 读取CSV文件
csv_file = '/root/autodl-tmp/Attenphos/Human dataset/ptm_data.csv'
df = pd.read_csv(csv_file)

# 创建输出目录
output_dir = 'ptm_fasta_files'
os.makedirs(output_dir, exist_ok=True)

# 获取所有不同的PTM类型
ptm_types = df['ptm'].unique()

# 遍历每种PTM类型并生成相应的FASTA文件
for ptm_type in ptm_types:
    # 过滤当前PTM类型的数据
    ptm_df = df[df['ptm'] == ptm_type]

    # 构建FASTA文件路径
    fasta_file_path = os.path.join(output_dir, f'{ptm_type}.fasta')

    # 写入FASTA文件
    with open(fasta_file_path, 'w') as fasta_file:
        for _, row in ptm_df.iterrows():
            # 提取FASTA标题和完整trunc_seq序列
            header = f'>{row["entry"]}|{row["pos"]}'
            sequence = row["trunc_seq"]  # 使用完整的trunc_seq
            fasta_file.write(f'{header}\n{sequence}\n')

print("FASTA files generated successfully.")