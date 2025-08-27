import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dense, Embedding, Dropout, LayerNormalization
from tensorflow.keras.models import Model
import json

# 配置GPU内存增长
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# ==================== 配置类 ====================
class Config:
    def __init__(self, ptm_type='phosphorylation'):
               # 修改目标氨基酸映射
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
        self.dataset_name = "ptm_data"
        self.ptm_type = ptm_type.lower()
        self.target_aa = self.ptm_aa_map.get(self.ptm_type, [])
        
        if not self.target_aa:
            raise ValueError(f"Unsupported PTM type: {ptm_type}")

        self.check_esm_validity = True  # 是否检查ESM特征有效性
        self.filter_invalid_esm = False  # 是否过滤无效ESM样本
        # 路径配置 
        self.root_dir  = "/root/autodl-tmp/Attenphos"
        self.esm_dir  = "/root/autodl-tmp/Attenphos/kk/train_PTM"
        self.esm2  = "/root/autodl-tmp/Attenphos/kk"
        self.result_dir  = os.path.join(self.esm_dir,  f"results_{self.ptm_type}_esm2_kfold") 
        self.model_dir  = os.path.join(self.esm_dir,  f"models_{self.ptm_type}_esm2_kfold") 
        self.data_dir  = os.path.join(self.root_dir,  "Human dataset")
        
        # ESM特征文件路径 (根据您的实际情况修改)
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
        
        # 训练参数 
        self.batch_size  = 64 
        self.initial_learning_rate  = 0.0005 
        self.epochs  =200
        self.patience  =39
        self.warmup_epochs  = 5 
        
        # 数据参数 
        self.window_sizes  = [21, 33, 51]
        self.max_features  = 21 
        self.empty_aa  = '*'
        self.test_size  = 0.2 
        # K折交叉验证参数
        self.k_folds = 10  # 使用5折交叉验证 
        self.random_state  = 42  # 随机种子
        # ESM配置 
        self.use_esm  = True 
        self.esm_dim  = 1280  # ESM2-t33_650M_UR50D的特征维度 
        
        # 结构特征配置 
        self.struct_features  = ['sasa', 'phi', 'psi', 'secstruct', 'local_plDDT', 'avg_plDDT']
        self.struct_feature_dim  = 8  # 将在初始化时计算 
# ==================== 自定义层定义 ====================
class PositionEmbedding(Layer):
    def __init__(self, max_len=100, embed_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.pos_emb = self.add_weight(
            name='position_embedding',
            shape=(self.max_len, self.embed_dim),
            initializer='glorot_uniform',
            trainable=True)
        super().build(input_shape)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = tf.cast(positions, tf.int32)
        return x + tf.nn.embedding_lookup(self.pos_emb, positions)

    def get_config(self):
        config = super().get_config()
        config.update({
            'max_len': self.max_len,
            'embed_dim': self.embed_dim
        })
        return config

class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim

    def build(self, input_shape):
        self.query = Dense(self.embed_dim)
        self.key = Dense(self.embed_dim)
        self.value = Dense(self.embed_dim)
        self.combine = Dense(self.embed_dim)
        super().build(input_shape)

    def call(self, x):
        batch_size = tf.shape(x)[0]

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = tf.reshape(q, (batch_size, -1, self.num_heads, self.head_dim))
        q = tf.transpose(q, [0, 2, 1, 3])

        k = tf.reshape(k, (batch_size, -1, self.num_heads, self.head_dim))
        k = tf.transpose(k, [0, 2, 1, 3])

        v = tf.reshape(v, (batch_size, -1, self.num_heads, self.head_dim))
        v = tf.transpose(v, [0, 2, 1, 3])

        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention = tf.nn.softmax(matmul_qk / tf.math.sqrt(dk), axis=-1)
        output = tf.matmul(scaled_attention, v)

        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.embed_dim))
        return self.combine(output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads
        })
        return config

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='swish'),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=True):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate
        })
        return config

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
            
            print(f"\n=== ESM特征统计 ===")
            print(f"总特征数: {len(features)}")
            valid_features = np.any(features != 0, axis=1).sum()
            print(f"有效特征数: {valid_features} ({valid_features/len(features):.1%})")
            
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


# ==================== 特征提取工具 ====================
def create_feature_extractor(model, layer_names=None):
    """创建特征提取模型"""
    if layer_names is None:
        # 默认提取所有密集层和最后的Transformer块
        layer_names = []
        for layer in model.layers:
            if isinstance(layer, (Dense, TransformerBlock)):
                layer_names.append(layer.name)
    
    # 获取指定层的输出
    layer_outputs = [layer.output for layer in model.layers if layer.name in layer_names]
    
    # 如果没有指定层，则使用倒数第二层（通常是分类层之前的层）
    if not layer_outputs:
        layer_outputs = [model.layers[-2].output]
    
    # 创建特征提取模型
    feature_model = tf.keras.Model(inputs=model.inputs, outputs=layer_outputs)
    return feature_model

def extract_features(feature_model, X_data, batch_size=64):
    """提取特征"""
    features = feature_model.predict(X_data, batch_size=batch_size, verbose=1)
    return features
# ==================== 样本平衡处理 ====================
def balance_samples(features, labels, ptm_type, N=3000, M=2500, random_state=42):
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler

    # 分离正负样本索引
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    print(f"\n原始样本分布 - 阳性: {len(pos_idx)}, 阴性: {len(neg_idx)}")
#RandomOverSampler  RandomUnderSampler
    # 对正样本进行过/欠采样至M个
    if len(pos_idx) > M:
        ros = RandomUnderSampler(sampling_strategy={1: M}, random_state=random_state)
        features, labels = ros.fit_resample(features, labels)
    
    # 对负样本进行过/欠采样至N个
    if len(neg_idx) > N:
        rus = RandomUnderSampler(sampling_strategy={0: N}, random_state=random_state)
        features, labels = rus.fit_resample(features, labels)

    # 打印最终样本分布
    new_pos = np.sum(labels == 1)
    new_neg = np.sum(labels == 0)
    print(f"平衡后样本分布 - 阳性: {new_pos}, 阴性: {new_neg}")

    return features, labels
# ==================== t-SNE可视化 ====================
def plot_tsne(features, labels, ptm_type, perplexity=30, n_iter=1000, random_state=42):
    """执行t-SNE降维并可视化"""
    # 根据PTM类型进行样本平衡 
    balanced_features, balanced_labels = balance_samples(
        features, labels, 
        ptm_type=ptm_type,
        random_state=random_state 
    )
    
    # 标准化特征 
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(balanced_features) 
 
    # 执行t-SNE 
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, 
                random_state=random_state, n_jobs=-1)
    tsne_results = tsne.fit_transform(features_scaled) 

    # 创建可视化 
    plt.figure(figsize=(12,  10))
    # scatter = sns.scatterplot( 
    #     x=tsne_results[:, 0], y=tsne_results[:, 1],
    #     hue=balanced_labels, palette=sns.color_palette("hls",  2),
    #     alpha=0.7, s=50, edgecolor='w', linewidth=0.5 
    # )
    # scatter = sns.scatterplot(
    #     x=tsne_results[:, 0], y=tsne_results[:, 1],
    #     hue=balanced_labels, palette=sns.color_palette("pastel", 2),  # 使用浅色调色板
    #     alpha=0.7, s=30, edgecolor='none'  # 减小点的大小和透明度
    # )
 
    # title = f't-SNE Visualization of {ptm_type.capitalize()}  PTM Sites'
    # plt.title(title,  fontsize=16, pad=20)
    # plt.xlabel('t-SNE  Dimension 1', fontsize=14)
    # plt.ylabel('t-SNE  Dimension 2', fontsize=14)
 
    # # 设置图例 
    # handles, _ = scatter.get_legend_handles_labels() 
    # plt.legend(handles,  ['Negative', 'Positive'], title='PTM Label',
    #            bbox_to_anchor=(1.05, 1), loc='upper left')
 
    # plt.grid(True,  linestyle='--', alpha=0.3)
    colors = ['#FFA500' if label == 1 else '#000080' for label in balanced_labels ]
    
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
    red_patch = mpatches.Patch(color='#FFA500', label='Positive')
    blue_patch = mpatches.Patch(color='#000080', label='Negative')
    plt.legend(handles=[red_patch, blue_patch], 
               title='PTM Label',
               bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.grid(True, linestyle='--', alpha=0.3)
 
    # 保存图像 
    output_dir = "/root/autodl-tmp/Attenphos/kk/train_PTM/t-sne/phosphorylation_after"
    os.makedirs(output_dir,  exist_ok=True)
    filename = f"tsne_{ptm_type}_p{perplexity}_i{n_iter}.png"
    save_path = os.path.join(output_dir,  filename)
    plt.savefig(save_path,  dpi=300, bbox_inches='tight')
    plt.close() 
 
    print(f"t-SNE图已保存至: {save_path}")
    return tsne_results 

# ==================== 主程序 ====================
def main(ptm_type='phosphorylation'):
    # 配置
    config = Config(ptm_type=ptm_type)
    data_loader = PTMDataLoader(config)
    
    # 加载数据
    sequences, positions, labels, protein_ids, struct_features, esm_features = data_loader.load_dataset()
    print(f"\n=== 数据统计 ===")
    print(f"总样本数: {len(labels)}")
    print(f"阳性样本数: {np.sum(labels == 1)}")
    print(f"阴性样本数: {np.sum(labels == 0)}")
    
    # 准备多尺度序列特征
    X_seq = []
    for win_size in config.window_sizes:
        seq_windows = data_loader.prepare_samples(sequences, positions, win_size)
        X_seq.append(data_loader.encode_sequences(seq_windows))
    
    # 合并所有特征
    X_all = X_seq + [struct_features, esm_features]
    y_all = labels
    
    # 加载模型
    model_path = os.path.join(config.model_dir, f"ptm_data_301_39_64_fold_10_best_model.h5")
    try:
        custom_objects = {
            'PositionEmbedding': PositionEmbedding,
            'MultiHeadSelfAttention': MultiHeadSelfAttention,
            'TransformerBlock': TransformerBlock
        }
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        print("\n模型加载成功!")
    except Exception as e:
        print(f"\n加载模型时出错: {str(e)}")
        return
    
    # 创建特征提取器
    print("\n创建特征提取器...")
    feature_extractor = create_feature_extractor(model)
    
    # 提取特征
    print("\n正在提取特征...")
    features = extract_features(feature_extractor, X_all)
    
    # 如果features是列表，选择最后一个Dense层的特征进行可视化
    if isinstance(features, list):
        last_layer_features = features[-1]
    else:
        last_layer_features = features
    
    print(f"\n特征形状: {last_layer_features.shape}")
    
    # 执行t-SNE可视化
    print("\n正在执行t-SNE降维...")
    
    # 使用不同参数进行多次可视化 
    for perplexity in [1.5,1,2]:
        for n_iter in [500, 1000]:
            tsne_results = plot_tsne(
                last_layer_features,
                y_all,
                ptm_type=ptm_type,
                perplexity=perplexity,
                n_iter=n_iter 
            )
        
    # 保存t-SNE结果
    output_dir = "/root/autodl-tmp/Attenphos/kk/train_PTM/t-sne/phosphorylation_after"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"tsne_results_{ptm_type}.npy"), tsne_results)
    np.save(os.path.join(output_dir, f"tsne_labels_{ptm_type}.npy"), y_all)
    
    print("\nt-SNE可视化完成!")

if __name__ == "__main__":

    PTM_TYPES = ['phosphorylation']
    
    for ptm in PTM_TYPES:
        print(f"\n{'='*50}")
        print(f"正在处理: {ptm.upper()} PTM预测")
        print(f"{'='*50}")
        main(ptm_type=ptm)