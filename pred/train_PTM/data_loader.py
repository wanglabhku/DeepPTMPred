import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import train_test_split
import json

# ==================== 数据加载与处理 ====================
class PTMDataLoader:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        
        # 定义PTM类型到修饰氨基酸的映射
        self.ptm_aa_map = {
            'phosphorylation': ['S', 'T'],
            'acetylation': ['K'],
            'ubiquitination': ['K'],
            'hydroxylation': ['P'],
            'gamma_carboxyglutamic_acid': ['E'],
            'lys_methylation': ['K'],
            'malonylation': ['K'],
            'arg_methylation': ['R'],
            'crotonylation': ['K'],
            'succinylation': ['K'],
            'glutathionylation': ['C'],
            'sumoylation': ['K'],
            's_nitrosylation': ['C'],
            'glutarylation': ['K'],
            'citrullination': ['R'],
            'o_linked_glycosylation': ['S', 'T'],
            'n_linked_glycosylation': ['N']
        }
        
        # 定义PTM类型名称匹配模式
        self.ptm_patterns = {
            'phosphorylation': r'phospho|Phospho',
            'acetylation': r'acety|Acety',
            'ubiquitination': r'ubiquit|Ubiquit',
            'hydroxylation': r'hydroxy|Hydroxy',
            'gamma_carboxyglutamic_acid': r'gamma.*carboxy|Gamma.*Carboxy',
            
            'lys_methylation': r'lys.*methyl|Lys.*Methyl',
            'malonylation': r'malonyl|Malonyl',
            'arg_methylation': r'arg.*methyl|Arg.*Methyl',
            'crotonylation': r'crotonyl|Crotonyl',
            'succinylation': r'succinyl|Succinyl',
            'glutathionylation': r'glutathion|Glutathion',
            'sumoylation': r'sumoy|Sumoy',
            's_nitrosylation': r'nitrosyl|Nitrosyl',
            'glutarylation': r'glutaryl|Glutaryl',
            'citrullination': r'citrullin|Citrullin',
            'o_linked_glycosylation': r'O-.*glycos|o-.*glycos',
            'n_linked_glycosylation': r'N-.*glycos|n-.*glycos'
        }
        
    def load_esm_features(self):
        """加载ESM特征并与主数据对齐"""
        try:
            features = np.load(self.config.esm_feature_path)
            meta = np.load(self.config.esm_meta_path, allow_pickle=True)
            

            valid_features = np.any(features != 0, axis=1).sum()
            
            esm_dict = {}
            for i, (protein_id, pos, is_valid) in enumerate(zip(
                meta['protein_ids'], 
                meta['positions'], 
                meta['is_valid']
            )):
                clean_id = protein_id.split('-')[0]
                key = f"{clean_id}_{pos}"
                esm_dict[key] = {
                    'features': features[i],
                    'is_valid': is_valid,
                    'mod_aa': meta['mod_aas'][i]
                }
            
            return esm_dict
            
        except Exception as e:
            print(f"加载ESM特征时出错: {str(e)}")
            print("将使用零向量作为替代")
            return None
    
    def filter_ptm_data(self, df, ptm_type):
        """根据PTM类型筛选数据"""
        target_aa = self.ptm_aa_map.get(ptm_type, [])
        if not target_aa:
            raise ValueError(f"Unsupported PTM type: {ptm_type}")
        
        pattern = self.ptm_patterns.get(ptm_type, '')
        if not pattern:
            raise ValueError(f"No matching pattern for PTM type: {ptm_type}")
        
        # 筛选PTM类型
        filtered = df[df['ptm'].str.contains(pattern, case=False, regex=True)]
        
        # 筛选目标氨基酸
        filtered = filtered[filtered['mod_aa'].isin(target_aa)]
        
        # 检查数据有效性
        if len(filtered) == 0:
            print(f"警告: 未找到 {ptm_type} 类型的有效数据")
            return None
        
        print(f"找到 {len(filtered)} 条 {ptm_type} 记录")
        print(f"氨基酸分布: {filtered['mod_aa'].value_counts().to_dict()}")
        
        return filtered
    
    def load_dataset(self):
        """加载混合PTM数据集并自动分离不同类型"""
        data_path = os.path.join(self.config.data_dir, f"{self.config.dataset_name}.csv") 
        df = pd.read_csv(data_path) 
        
        # 根据当前PTM类型筛选数据
        df = self.filter_ptm_data(df, self.config.ptm_type)
        if df is None:
            raise ValueError(f"没有找到 {self.config.ptm_type} 的有效数据")
        
        # 验证位置是否有效
        invalid_pos = df.apply(lambda x: x['pos'] < 1 or x['pos'] > len(x['trunc_seq']), axis=1)
        if invalid_pos.any(): 
            print(f"发现 {invalid_pos.sum()} 条记录有无效位置")
            df = df[~invalid_pos]
        
        # 处理结构特征
        df['phi'] = df['phi'].apply(lambda x: np.array(eval(x))) 
        df['psi'] = df['psi'].apply(lambda x: np.array(eval(x))) 
        
        # 提取中心位置特征
        half_window = (max(self.config.window_sizes) - 1) // 2 
        df['phi_center'] = df['phi'].apply(lambda x: x[half_window] if len(x) > half_window else 0.0)
        df['psi_center'] = df['psi'].apply(lambda x: x[half_window] if len(x) > half_window else 0.0)
        
        # 构建特征矩阵
        struct_features = np.column_stack([ 
            df['sasa'].values,
            df['phi_center'].values,
            df['psi_center'].values,
            pd.get_dummies(df['secstruct']).values, 
            df['local_plDDT'].values,
            df['avg_plDDT'].values 
        ])
        
        # 处理ESM特征
        esm_dict = self.load_esm_features()
        esm_features = []
        valid_flags = []
        mismatch_log = []
    
        for _, row in df.iterrows():
            protein_id = row['entry']
            pos = row['pos']
            mod_aa = row['mod_aa']
            lookup_key = f"{protein_id}_{pos}"
            
            if esm_dict is not None and lookup_key in esm_dict:
                esm_mod_aa = esm_dict[lookup_key]['mod_aa']
                if esm_mod_aa != mod_aa:
                    mismatch_log.append(f"{lookup_key}: ESM修饰类型不匹配(ESM:{esm_mod_aa}, 当前:{mod_aa})")
                
                if esm_dict[lookup_key]['is_valid']:
                    esm_features.append(esm_dict[lookup_key]['features'])
                    valid_flags.append(True)
                    continue
            
            # 无效或缺失的特征
            esm_features.append(np.zeros(self.config.esm_dim))
            valid_flags.append(False)
        
        # 打印对齐检查结果
        if mismatch_log:
            print("\n警告: 发现ESM修饰类型不匹配样本 (前5个):")
            for msg in mismatch_log[:5]:
                print(msg)
            print(f"共发现 {len(mismatch_log)} 处修饰类型不匹配")
        
        esm_features = np.array(esm_features) 
        valid_flags = np.array(valid_flags) 
        
        # 统计无效样本
        invalid_count = len(valid_flags) - np.sum(valid_flags) 
        if invalid_count > 0:
            print(f"警告: 发现 {invalid_count} 个无效ESM特征样本")
        
        return (df['trunc_seq'].values,
                df['pos'].values,
                df['label'].values,
                df['entry'].values,
                struct_features,
                esm_features)
    
    def prepare_samples(self, sequences, positions, window_size):
        """准备序列窗口样本"""
        half_len = (window_size - 1) // 2 
        encoded_seqs = []
        
        for seq, pos in zip(sequences, positions):
            if pos < 1 or pos > len(seq):
                print(f"警告: 无效位置 {pos} (序列长度: {len(seq)})")
                continue 
                
            center = seq[pos-1]  # 假设位置是1-based索引
            left = seq[max(0, pos-1-half_len):pos-1]
            right = seq[pos:min(len(seq), pos+half_len)]
            
            # 填充
            left_pad = max(0, half_len - len(left))
            right_pad = max(0, half_len - len(right))
            
            window = self.config.empty_aa * left_pad + left + center + right + self.config.empty_aa * right_pad 
            encoded_seqs.append(window) 
        
        return encoded_seqs 
    
    def encode_sequences(self, sequences):
        """将氨基酸序列编码为数字矩阵"""
        letter_dict = {
            "A":0, "C":1, "D":2, "E":3, "F":4, "G":5, "H":6, "I":7, "K":8, "L":9,
            "M":10, "N":11, "P":12, "Q":13, "R":14, "S":15, "T":16, "V":17, "W":18, "Y":19,
            "*":20 
        }
        
        encoded = np.zeros((len(sequences), len(sequences[0])), dtype=int)
        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq):
                encoded[i, j] = letter_dict.get(aa, 20)
        
        return encoded
