import os
import numpy as np
import torch
import argparse  # 必须导入
from esm import pretrained
from tqdm import tqdm

# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
custom_checkpoint_path = "/root/autodl-tmp/deepPTMpred/esm/hub/checkpoints/esm2_t33_650M_UR50D.pt"
os.makedirs("/root/autodl-tmp/deepPTMpred/esm", exist_ok=True)
os.environ["TORCH_HOME"] = "/root/autodl-tmp/deepPTMpred/esm"

def extract_full_sequence_esm(protein_id, sequence, output_dir="."):
    """
    提取整个蛋白质序列的ESM特征
    """
    # 定义模型路径
    custom_checkpoint_path = "/root/autodl-tmp/deepPTMpred/esm/hub/checkpoints/esm2_t33_650M_UR50D.pt"

    # 使用 safe_globals 安全加载模型
    with torch.serialization.safe_globals([argparse.Namespace]):
        model, alphabet = pretrained.load_model_and_alphabet(custom_checkpoint_path)

    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    
    # 准备输出文件
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{protein_id}_full_esm.npz")
    
    # 分割长序列（ESM最大支持1024 tokens）
    max_len = 1022  # 留出CLS和SEP位置
    sequences = [sequence[i:i+max_len] for i in range(0, len(sequence), max_len)]
    
    # 存储所有特征
    full_features = np.zeros((len(sequence), 1280))  # ESM2特征维度为1280
    
    for i, sub_seq in enumerate(tqdm(sequences, desc="处理序列片段")):
        try:
            # 转换为模型输入
            data = [(f"{protein_id}_part{i+1}", sub_seq)]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)
            
            # 提取特征 (移除CLS和SEP tokens)
            with torch.no_grad():
                results = model(tokens, repr_layers=[33])
                features = results["representations"][33][0, 1:-1].cpu().numpy()
            
            # 填充到完整数组
            start = i * max_len
            end = start + len(sub_seq)
            full_features[start:end] = features[:len(sub_seq)]
            
        except Exception as e:
            print(f"片段 {i+1} 处理失败: {str(e)}")
    
    # 保存结果
    np.savez_compressed(
        output_file,
        features=full_features,
        protein_id=protein_id,
        sequence=sequence,
        length=len(sequence))
    
    print(f"\n全序列ESM特征已保存到: {output_file}")
    print(f"特征矩阵形状: {full_features.shape}")  # 应为 (序列长度, 1280)

if __name__ == "__main__":
    
    protein_id = "P60484"
    protein_sequence = "MTAIIKEIVSRNKRRYQEDGFDLDLTYIYPNIIAMGFPAERLEGVYRNNIDDVVRFLDSKHKNHYKIYNLCAERHYDTAKFNCRVAQYPFEDHNPPQLELIKPFCEDLDQWLSEDDNHVAAIHCKAGKGRTGVMICAYLLHRGKFLKAQEALDFYGEVRTRDKKGVTIPSQRRYVYYYSYLLKNHLDYRPVALLFHKMMFETIPMFSGGTCNPQFVVCQLKVKIYSSNSGPTRREDKFMYFEFPQPLPVCGDIKVEFFHKQNKMLKKDKMFHFWVNTFFIPGPEETSEKVENGSLCDQEIDSICSIERADNDKEYLVLTLTKNDLDKANKDKANRYFSPNFKVKLYFTKTVEEPSNPEASSSTSVTPDVSDNEPDHYRYSDTTDSDPENEPFDEDQHTQITKV"
    
    extract_full_sequence_esm(
        protein_id=protein_id,
        sequence=protein_sequence,
        output_dir="/root/autodl-tmp/deepPTMpred/pred/custom_esm"
    )
