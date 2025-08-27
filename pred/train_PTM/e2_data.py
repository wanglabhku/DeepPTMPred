import os
import pandas as pd
import numpy as np
import torch
from esm import pretrained, Alphabet
import argparse
import traceback
from tqdm import tqdm

# 配置
custom_cache_path = "/root/autodl-tmp/Attenphos/esm"
os.environ["TORCH_HOME"] = custom_cache_path
os.makedirs(custom_cache_path, exist_ok=True)
def strict_sequence_processing(sequence, position, alphabet, max_len=1022):
    """处理序列，确保符合ESM输入要求"""
    valid_chars = set(alphabet.tok_to_idx.keys())
    clean_seq = ''.join([c for c in sequence if c in valid_chars])
    
    pos_idx = position - 1  # 转换为0-based索引
    half_len = max_len // 2
    
    # 提取中心位置周围的序列
    start = max(0, pos_idx - half_len)
    end = min(len(clean_seq), pos_idx + half_len + 1)  # +1确保包含中心位置
    
    # 调整窗口位置
    if (end - start) < max_len:
        if start == 0:
            end = min(len(clean_seq), max_len)
        else:
            start = max(0, len(clean_seq) - max_len)
    
    sub_seq = clean_seq[start:end][:max_len]
    
    # 填充不足长度的序列
    if len(sub_seq) < max_len:
        pad_left = max(0, half_len - pos_idx)
        pad_right = max_len - len(sub_seq) - pad_left
        sub_seq = ('X' * pad_left) + sub_seq + ('X' * pad_right)
    
    return sub_seq[:max_len], start  # 返回处理后的序列和原始起始位置

def load_data(file_path):
    """加载CSV数据"""
    df = pd.read_csv(file_path)
    # 确保必要的列存在
    required_cols = ['entry', 'pos', 'trunc_seq', 'label', 'mod_aa']
    assert all(col in df.columns for col in required_cols), "CSV缺少必要列"
    return df

def process_all_sequences(df, alphabet):
    """处理所有序列并记录问题样本"""
    processed = []
    error_log = []
    
    for _, row in df.iterrows():
        try:
            protein_id = row['entry']
            pos = int(row['pos'])
            seq = row['trunc_seq']
            mod_aa = row['mod_aa']
            
            # 验证修饰位点氨基酸
            if seq[pos-1] != mod_aa:
                raise ValueError(f"氨基酸不匹配(标注:{mod_aa}, 实际:{seq[pos-1]})")
                
            proc_seq, start_pos = strict_sequence_processing(seq, pos, alphabet)
            assert len(proc_seq) <= 1022, f"序列过长: {len(proc_seq)}"
            
            processed.append({
                'protein_id': protein_id,
                'sequence': proc_seq,
                'label': row['label'],
                'original_pos': pos,
                'window_start': start_pos,
                'mod_aa': mod_aa,
                'is_valid': True  # 标记有效样本
            })
        except Exception as e:
            error_log.append({
                'protein_id': row['entry'],
                'position': pos,
                'error': str(e)
            })
            # 仍然保留问题样本但标记为无效
            processed.append({
                'protein_id': row['entry'],
                'sequence': 'X' * 1022,  # 填充无效序列
                'label': row['label'],
                'original_pos': pos,
                'window_start': 0,
                'mod_aa': row['mod_aa'],
                'is_valid': False
            })
    
    # 打印错误统计
    if error_log:
        print("\n序列处理阶段发现问题样本:")
        error_df = pd.DataFrame(error_log)
        print(error_df['error'].value_counts())
    
    return processed, error_log

def extract_esm_features(sequences, alphabet, output_path, batch_size=64, resume=False):
    """提取ESM特征（自动跳过问题样本并用零向量填充）"""
    model_path = f"{custom_cache_path}/hub/checkpoints/esm2_t33_650M_UR50D.pt"
    temp_feature_file = output_path + ".temp.npy"
    temp_meta_file = output_path + ".temp_meta.npz"
    progress_file = output_path + ".progress.txt"

    # 加载模型
    with torch.serialization.safe_globals([argparse.Namespace]):
        model, _ = pretrained.load_model_and_alphabet(model_path)
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    # 初始化变量
    features = []
    metadata = {
        'protein_ids': [],
        'labels': [],
        'positions': [],
        'window_starts': [],
        'mod_aas': [],
        'is_valid': []  # 记录有效性标记
    }
    start_idx = 0

    # 恢复模式处理
    if resume and os.path.exists(temp_feature_file) and os.path.exists(temp_meta_file) and os.path.exists(progress_file):
        try:
            features = list(np.load(temp_feature_file, allow_pickle=True))
            saved_meta = np.load(temp_meta_file, allow_pickle=True)
            
            for key in metadata.keys():
                metadata[key] = list(saved_meta[key])
                
            with open(progress_file, 'r') as f:
                start_idx = int(f.read().strip())

            print(f"恢复进度: 已处理 {start_idx}/{len(sequences)} 个样本")
            
            if len(features) != start_idx:
                print("警告: 特征数量与进度记录不一致，将从头开始")
                features = []
                metadata = {k: [] for k in metadata.keys()}
                start_idx = 0

        except Exception as e:
            print(f"恢复失败: {str(e)}")
            features = []
            metadata = {k: [] for k in metadata.keys()}
            start_idx = 0

    # 进度条
    pbar = tqdm(total=len(sequences), initial=start_idx, desc="提取特征")

    try:
        for i in range(start_idx, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            
            # 分离有效和无效样本
            valid_samples = []
            valid_indices = []
            for idx, item in enumerate(batch):
                if item['is_valid']:
                    valid_samples.append((item['protein_id'], item['sequence']))
                    valid_indices.append(idx)
            
            # 处理有效样本
            if valid_samples:
                _, _, tokens = batch_converter(valid_samples)
                tokens = tokens.to(device)
                
                with torch.no_grad():
                    results = model(tokens, repr_layers=[33])
                    batch_features = results["representations"][33].mean(1).cpu().numpy()
            else:
                batch_features = np.zeros((0, 1280))  # ESM2特征维度
            
            # 构建完整批次特征（包含无效样本的零填充）
            full_batch_features = np.zeros((len(batch), 1280))
            for feat_idx, data_idx in enumerate(valid_indices):
                full_batch_features[data_idx] = batch_features[feat_idx]
            
            features.extend(full_batch_features)
            
            # 保存元数据
            for item in batch:
                metadata['protein_ids'].append(item['protein_id'])
                metadata['labels'].append(item['label'])
                metadata['positions'].append(item['original_pos'])
                metadata['window_starts'].append(item['window_start'])
                metadata['mod_aas'].append(item['mod_aa'])
                metadata['is_valid'].append(item['is_valid'])

            # 更新进度
            current_progress = i + len(batch)

            # 保存临时结果
            np.save(temp_feature_file, np.array(features))
            np.savez(temp_meta_file, **metadata)
            with open(progress_file, 'w') as f:
                f.write(str(current_progress))

            # 更新进度条
            pbar.update(len(batch))
            pbar.set_postfix({
                '批次': f"{(i // batch_size) + 1}/{(len(sequences) + batch_size - 1) // batch_size}",
                '有效样本': f"{len(valid_samples)}/{len(batch)}",
                'GPU内存': f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB"
            })

            torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\n用户中断: 已保存临时结果")
    except Exception as e:
        print(f"\n处理出错: {str(e)}")
        traceback.print_exc()
    finally:
        pbar.close()

        # 完成时保存最终结果
        if len(features) == len(sequences):
            np.save(output_path, np.array(features))
            np.savez(output_path.replace('_features_', '_metadata_'), **metadata)
            
            # 清理临时文件
            for f in [temp_feature_file, temp_meta_file, progress_file]:
                if os.path.exists(f):
                    os.remove(f)
            print(f"\n特征提取完成! 结果已保存到 {output_path}")

    return np.array(features), metadata

if __name__ == "__main__":
    # 输入输出路径
    input_csv = "/root/autodl-tmp/Attenphos/Human dataset/ptm_data.csv"
    output_feature = "/root/autodl-tmp/Attenphos/kk/ptm_data_esm_features.npy"
    output_metadata = "/root/autodl-tmp/Attenphos/kk/ptm_data_metadata.npz"

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="继续未完成的处理")
    parser.add_argument("--overwrite", action="store_true", help="强制重新开始")
    args = parser.parse_args()

    # # 加载模型和字母表
    # with torch.serialization.safe_globals([argparse.Namespace]):
    #     model, alphabet = pretrained.load_model_and_alphabet(
    #         f"{custom_cache_path}/hub/checkpoints/esm2_t33_650M_UR50D.pt"
    #     )
        # 加载模型和字母表
    print("\n正在加载ESM模型...")
    model_path = f"{custom_cache_path}/hub/checkpoints/esm2_t33_650M_UR50D.pt"
    with torch.serialization.safe_globals([argparse.Namespace]):
        model, alphabet = pretrained.load_model_and_alphabet(model_path)
    
    # 打印模型信息（现在model已定义）
    print("\n模型结构:")
    print(model)
    
    # 统计总参数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_params = count_parameters(model)
    print(f"\n模型总参数量: {total_params:,} (约 {total_params / 1e6:.2f} M)")
    
    # 检查GPU可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    # 处理数据
    df = load_data(input_csv)
    processed_data, error_log = process_all_sequences(df, alphabet)

    print("\n样本检查 (前5个):")
    for i, item in enumerate(processed_data[:5]):
        status = "有效" if item['is_valid'] else "无效"
        print(f"{i+1}. {item['protein_id']} 位置:{item['original_pos']} "
              f"修饰:{item['mod_aa']} 状态:{status}")

    print("\n开始特征提取...")
    features, metadata = extract_esm_features(
        processed_data,
        alphabet,
        output_feature,
        resume=args.resume and not args.overwrite
    )

    # 保存错误日志
    if error_log:
        error_log_path = output_metadata.replace(".npz", "_errors.csv")
        pd.DataFrame(error_log).to_csv(error_log_path, index=False)
        print(f"\n错误日志已保存至: {error_log_path}")