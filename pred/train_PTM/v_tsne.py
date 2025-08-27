import os 
import numpy as np 
import pandas as pd 
import tensorflow as tf 
from tensorflow.keras.models  import load_model 
from sklearn.manifold  import TSNE 
import matplotlib.pyplot  as plt 
import seaborn as sns 
from sklearn.preprocessing  import StandardScaler 
from tensorflow.keras  import backend as K 
from tensorflow.keras.layers  import Layer, Dense, Embedding, Dropout, LayerNormalization 
from tensorflow.keras.models  import Model 
import json 
from tensorflow.keras.layers  import Input, Embedding 
# 配置GPU内存增长 
gpus = tf.config.experimental.list_physical_devices('GPU') 
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,  True)
    except RuntimeError as e:
        print(e)
 
# ==================== 配置类 ====================
class Config:
    def __init__(self, ptm_type='phosphorylation'):
        self.ptm_aa_map  = {
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
        self.dataset_name  = "ptm_data"
        self.ptm_type  = ptm_type.lower() 
        self.target_aa  = self.ptm_aa_map.get(self.ptm_type,  [])
        
        if not self.target_aa: 
            raise ValueError(f"Unsupported PTM type: {ptm_type}")
 
        # 路径配置 
        self.root_dir  = "/root/autodl-tmp/Attenphos"
        self.esm_dir  = "/root/autodl-tmp/Attenphos/kk/train_PTM"
        self.esm2  = "/root/autodl-tmp/Attenphos/kk"
        self.result_dir  = os.path.join(self.esm_dir,  f"results_{self.ptm_type}_esm2_kfold")  
        self.model_dir  = os.path.join(self.esm_dir,  f"models_{self.ptm_type}_esm2_kfold")  
        self.data_dir  = os.path.join(self.root_dir,  "Human dataset")
        
        # ESM特征文件路径 
        self.esm_feature_path  = os.path.join(self.esm2,  "ptm_data_esm_features.npy")  
        self.esm_meta_path  = os.path.join(self.esm2,  "ptm_data_metadata.npz")  
        
        os.makedirs(self.result_dir,  exist_ok=True)
        os.makedirs(self.model_dir,  exist_ok=True)
        
        # 模型参数 
        self.embed_dim  = 64 
        self.num_heads  = 4 
        self.ff_dim  = 256 
        self.dropout_rate  = 0.3 
        self.weight_decay  = 0.0001 
        
        # 数据参数 
        self.window_sizes  = [21, 33, 51]
        self.max_features  = 21 
        self.empty_aa  = '*' 
        self.use_esm  = True 
        self.esm_dim  = 1280  # ESM2特征维度 
        
        # 结构特征配置 
        self.struct_features  = ['sasa', 'phi', 'psi', 'secstruct', 'local_plDDT', 'avg_plDDT']
        self.struct_feature_dim  = 8 
 
# ==================== 数据加载与处理 ====================
class PTMDataLoader:
    def __init__(self, config):
        self.config  = config 
        self.scaler  = StandardScaler()
        
    def load_dataset(self):
        """加载数据集"""
        data_path = os.path.join(self.config.data_dir,  f"{self.config.dataset_name}.csv")  
        df = pd.read_csv(data_path)  
        
        # 筛选当前PTM类型的数据 
        df = df[df['ptm'].str.contains(self.config.ptm_type,  case=False, regex=True)]
        df = df[df['mod_aa'].isin(self.config.target_aa)] 
        
        if len(df) == 0:
            raise ValueError(f"没有找到 {self.config.ptm_type}  的有效数据")
        
        # 处理结构特征 
        df['phi'] = df['phi'].apply(lambda x: np.array(eval(x)))  
        df['psi'] = df['psi'].apply(lambda x: np.array(eval(x)))  
        
        # 提取中心位置特征 
        half_window = (max(self.config.window_sizes)  - 1) // 2 
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
        
        # 加载ESM特征 
        esm_features = np.zeros((len(df),  self.config.esm_dim)) 
        try:
            esm_data = np.load(self.config.esm_feature_path) 
            if len(esm_data) == len(df):
                esm_features = esm_data 
        except:
            pass 
        
        return (df['trunc_seq'].values,
                df['pos'].values,
                df['label'].values,
                struct_features,
                esm_features)
    
    def prepare_samples(self, sequences, positions, labels, window_size):
        half_len = (window_size - 1) // 2 
        valid_seqs = []
        valid_labels = []
    
        for seq, pos, label in zip(sequences, positions, labels):
            if 1 <= pos <= len(seq):  # 有效位置检查 
                left = seq[max(0, pos-1-half_len):pos-1]
                right = seq[pos:min(len(seq), pos+half_len)]
                window = self.config.empty_aa * (half_len - len(left)) + left + seq[pos-1] + right 
                valid_seqs.append(window) 
                valid_labels.append(label) 
    
        return valid_seqs, np.array(valid_labels)
    
    def encode_sequences(self, sequences):
        """将氨基酸序列编码为数字矩阵"""
        letter_dict = {
            "A":0, "C":1, "D":2, "E":3, "F":4, "G":5, "H":6, "I":7, "K":8, "L":9,
            "M":10, "N":11, "P":12, "Q":13, "R":14, "S":15, "T":16, "V":17, "W":18, "Y":19,
            "*":20 
        }
        
        encoded = np.zeros((len(sequences),  len(sequences[0])), dtype=int)
        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq):
                encoded[i, j] = letter_dict.get(aa,  20)
        
        return encoded 
 
# ==================== 特征提取 ====================
def get_embedding_model(config):
    """创建嵌入模型用于提取嵌入特征"""
    # 序列输入 
    seq_input = Input(shape=(config.window_sizes[0],),  name='seq_input')
    
    # 嵌入层 
    x = Embedding(config.max_features,  config.embed_dim)(seq_input) 
    
    # 创建模型 
    model = Model(inputs=seq_input, outputs=x)
    return model 
 
# ==================== 样本平衡 ====================
def balance_samples(features, labels, random_state=42):
    """平衡样本数量"""
    from imblearn.under_sampling  import RandomUnderSampler 
    
    # 计算样本数量 
    pos_count = np.sum(labels  == 1)
    neg_count = np.sum(labels  == 0)
    min_count = min(pos_count, neg_count)
    
    # 如果样本已经平衡或数量不足，直接返回 
    if pos_count == neg_count or min_count < 100:
        return features, labels 
    
    # 使用随机欠采样 
    sampler = RandomUnderSampler(
        sampling_strategy={0: min_count, 1: min_count},
        random_state=random_state 
    )
    
    # 调整特征形状以适应采样器 
    if len(features.shape)  > 2:
        original_shape = features.shape  
        features_reshaped = features.reshape(features.shape[0],  -1)
        features_resampled, labels_resampled = sampler.fit_resample(features_reshaped,  labels)
        features_resampled = features_resampled.reshape(-1,  *original_shape[1:])
    else:
        features_resampled, labels_resampled = sampler.fit_resample(features,  labels)
    
    return features_resampled, labels_resampled 
 
# ==================== t-SNE可视化 ====================
def plot_tsne(features, labels, ptm_type, perplexity=30, n_iter=1000, random_state=42):
    """执行t-SNE降维并可视化"""
    # 平衡样本 
    features_balanced, labels_balanced = balance_samples(features, labels, random_state)

    # 如果特征是多维的，展平以进行t-SNE 
    if len(features_balanced.shape)  > 2:
        features_flat = features_balanced.reshape(features_balanced.shape[0],  -1)
    else:
        features_flat = features_balanced 
    
    # 标准化特征 
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_flat) 
    
    # 执行t-SNE 
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, 
                random_state=random_state, n_jobs=-1)
    tsne_results = tsne.fit_transform(features_scaled) 
    
    # 创建可视化 
    plt.figure(figsize=(12,  10))
    palette = sns.color_palette("husl",  2)
    
    # scatter = sns.scatterplot( 
    #     x=tsne_results[:, 0], y=tsne_results[:, 1],
    #     hue=labels_balanced, palette=palette,
    #     alpha=0.7, s=50, edgecolor='w', linewidth=0.5 
    # )
    # scatter = sns.scatterplot(
    #     x=tsne_results[:, 0], y=tsne_results[:, 1],
    #     hue=labels_balanced, palette=sns.color_palette("pastel", 2),  # 使用浅色调色板
    #     alpha=0.9, s=30, edgecolor='none'  # 减小点的大小和透明度
    # )
    colors = ['#FFFF00' if label == 1 else '#800080' for label in labels_balanced]
    
    scatter = plt.scatter(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        c=colors, alpha=0.6, s=50, edgecolor='k', linewidth=0.5
    )
    plt.title(f't-SNE  untrain Visualization of {ptm_type.capitalize()}  PTM Sites\n'
              )
    plt.xlabel('t-SNE  Dimension 1', fontsize=12)
    plt.ylabel('t-SNE  Dimension 2', fontsize=12)
    
    # 设置图例 
  
        # 创建自定义图例
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='#FFFF00', label='Positive')
    blue_patch = mpatches.Patch(color='#800080', label='Negative')
    plt.legend(handles=[red_patch, blue_patch], 
               title='PTM Label',
               bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 保存图像 
    output_dir = os.path.join("/root/autodl-tmp/Attenphos/esm2_feature/train_PTM/t-sne",  ptm_type)
    os.makedirs(output_dir,  exist_ok=True)
    filename = f"tsne_{ptm_type}_p{perplexity}_i{n_iter}.png"
    save_path = os.path.join(output_dir,  filename)
    plt.savefig(save_path,  dpi=300, bbox_inches='tight')
    plt.close() 
    
    print(f"t-SNE图已保存至: {save_path}")
    return tsne_results 
 
# ==================== 主程序 ====================
def main(ptm_type='ubiquitination'):
    config = Config(ptm_type=ptm_type)
    data_loader = PTMDataLoader(config)

    sequences, positions, labels, struct_features, esm_features = data_loader.load_dataset()
    print(f"\n=== 数据统计 ===")
    print(f"总样本数: {len(labels)}")
    print(f"阳性样本数: {np.sum(labels == 1)}")
    print(f"阴性样本数: {np.sum(labels == 0)}")

    # 使用 prepare_samples 同步获取序列和标签
    seq_windows, filtered_labels = data_loader.prepare_samples(
        sequences, positions, labels, config.window_sizes[0]
    )
    seq_encoded = data_loader.encode_sequences(seq_windows) 

    embedding_model = get_embedding_model(config)
    embedded_features = embedding_model.predict(seq_encoded, batch_size=64, verbose=1)
    print(f"嵌入特征形状: {embedded_features.shape}")

    # ✅ 关键修正：传入 filtered_labels 而不是原始 labels
    print("\n正在执行t-SNE降维...")
    perplexities = [5, 1.5, 2]
    iterations = [500, 1000]

    for perplexity in perplexities:
        for n_iter in iterations:
            print(f"\n参数: perplexity={perplexity}, iterations={n_iter}")
            tsne_results = plot_tsne(
                embedded_features,
                filtered_labels,  # ✅ 使用过滤后的标签
                ptm_type=ptm_type,
                perplexity=perplexity,
                n_iter=n_iter
            )

    print("\n所有t-SNE可视化完成!")
 
if __name__ == "__main__":
    # 选择要可视化的PTM类型 
    PTM_TYPES = [ 'phosphorylation']
    # PTM_TYPES = ['phosphorylation', 'acetylation', 'ubiquitination']
    
    for ptm in PTM_TYPES:
        print(f"\n{'='*50}")
        print(f"正在处理: {ptm.upper()}  PTM预测")
        print(f"{'='*50}")
        main(ptm_type=ptm)