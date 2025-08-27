import os 
import numpy as np 
import pandas as pd 
import tensorflow as tf 
import matplotlib.pyplot  as plt 
import seaborn as sns 
from tensorflow.keras.models  import load_model 
from tensorflow.keras  import backend as K 
from sklearn.inspection  import permutation_importance 
from sklearn.manifold  import TSNE 
from sklearn.decomposition  import PCA 
from Bio import motifs 
from Bio.Seq import Seq 
from Bio.SeqRecord import SeqRecord 
from tqdm import tqdm 
import logomaker 
import json 
from tensorflow_addons.optimizers  import AdamW 
from sklearn.preprocessing  import StandardScaler 
from sklearn.metrics  import (confusion_matrix, accuracy_score, f1_score, matthews_corrcoef,
                           roc_auc_score, precision_score, recall_score, 
                           precision_recall_curve, auc, average_precision_score,
                           classification_report)
from tensorflow.keras.layers  import Input, Embedding, Dense 
from tensorflow.keras.models  import Model 
from tensorflow_addons.losses  import SigmoidFocalCrossEntropy 
# 配置类 
class Config:
    def __init__(self, ptm_type='phosphorylation'):
        self.ptm_type  = ptm_type.lower()  
        if self.ptm_type  == 'phosphorylation':
            self.target_aa  = ['S', 'T']
        elif self.ptm_type  in ['acetylation', 'ubiquitination']:
            self.target_aa  = ['K']
        self.dataset_name  = "ptm_data"
        # 路径配置 
        self.root_dir  = "/root/autodl-tmp/Attenphos"
        self.esm_dir  = "/root/autodl-tmp/Attenphos/esm2_feature/train_PTM"
        self.esm2  = "/root/autodl-tmp/Attenphos/esm2_feature"
        self.result_dir  = os.path.join(self.esm_dir,  f"results_{self.ptm_type}_esm2_kfold")  
        self.model_dir  = os.path.join(self.esm_dir,  f"models_{self.ptm_type}_esm2_kfold")  
        self.data_dir  = os.path.join(self.root_dir,  "Human dataset")
        # ESM特征文件路径 
        self.esm_feature_path  = os.path.join(self.esm2,  "ptm_data_esm_features.npy")  
        self.esm_meta_path  = os.path.join(self.esm2,  "ptm_data_metadata.npz")  
        # 模型参数 
        self.embed_dim  = 64 
        self.num_heads  = 4 
        self.ff_dim  = 256 
        self.dropout_rate  = 0.3 
        self.weight_decay  = 0.0001 
        self.epochs  = 301
        self.patience  = 39 
        self.batch_size  = 64 
        # 数据参数 
        self.window_sizes  = [21, 33, 51]
        self.max_features  = 21 
        self.empty_aa  = '*'
        self.test_size  = 0.2 
        # ESM配置 
        self.use_esm  = True 
        self.esm_dim  = 1280 
        # 结构特征配置 
        self.struct_features  = ['sasa', 'phi', 'psi', 'secstruct', 'local_plDDT', 'avg_plDDT']
        self.struct_feature_dim  = 8 
 
class PTMDataLoader:
    def __init__(self, config):
        self.config  = config 
        self.scaler  = StandardScaler()
        
    def load_esm_features(self):
        """加载ESM特征并与主数据对齐"""
        try:
            features = np.load(self.config.esm_feature_path)  
            meta = np.load(self.config.esm_meta_path,  allow_pickle=True)
            
            print(f"\n=== ESM特征统计 ===")
            print(f"总特征数: {len(features)}")
            valid_features = np.any(features  != 0, axis=1).sum()
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
    
    def load_dataset(self):
        """加载混合PTM数据集并自动分离不同类型"""
        data_path = os.path.join(self.config.data_dir,  f"{self.config.dataset_name}.csv")   
        df = pd.read_csv(data_path)   
        
        # 根据当前PTM类型筛选数据 
        if self.config.ptm_type  == 'phosphorylation':
            df = df[df['ptm'].str.contains('Phosphorylation',  case=False)]
            df = df[df['mod_aa'].isin(['S', 'T'])]
        elif self.config.ptm_type  == 'acetylation':
            df = df[df['ptm'].str.contains('Acetylation',  case=False)]
            df = df[df['mod_aa'] == 'K']
        elif self.config.ptm_type  == 'ubiquitination':
            df = df[df['ptm'].str.contains('Ubiquitination',  case=False)]
            df = df[df['mod_aa'] == 'K']
        
        # 验证位置是否有效 
        invalid_pos = df.apply(lambda  x: x['pos'] < 1 or x['pos'] > len(x['trunc_seq']), axis=1)
        if invalid_pos.any():   
            print(f"发现 {invalid_pos.sum()}  条记录有无效位置")
            df = df[~invalid_pos]
        
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
        
        # 修改ESM特征处理部分 
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
                    mismatch_log.append(f"{lookup_key}:  ESM修饰类型不匹配(ESM:{esm_mod_aa}, 当前:{mod_aa})")
                
                if esm_dict[lookup_key]['is_valid']:
                    esm_features.append(esm_dict[lookup_key]['features'])  
                    valid_flags.append(True)  
                    continue 
            
            esm_features.append(np.zeros(self.config.esm_dim))  
            valid_flags.append(False)  
        
        if mismatch_log:
            print("\n警告: 发现ESM修饰类型不匹配样本 (前5个):")
            for msg in mismatch_log[:5]:
                print(msg)
            print(f"共发现 {len(mismatch_log)} 处修饰类型不匹配")
        
        esm_features = np.array(esm_features)   
        valid_flags = np.array(valid_flags)   
        
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
                
            center = seq[pos-1]
            left = seq[max(0, pos-1-half_len):pos-1]
            right = seq[pos:min(len(seq), pos+half_len)]
            
            left_pad = max(0, half_len - len(left))
            right_pad = max(0, half_len - len(right))
            
            window = self.config.empty_aa  * left_pad + left + center + right + self.config.empty_aa  * right_pad 
            encoded_seqs.append(window)   
        
        return encoded_seqs 
    
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
 
class PositionEmbedding(tf.keras.layers.Layer):  
    def __init__(self, max_len=100, embed_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.max_len  = max_len 
        self.embed_dim  = embed_dim 
 
    def build(self, input_shape):
        self.pos_emb  = self.add_weight(  
            name='position_embedding',
            shape=(self.max_len,  self.embed_dim),  
            initializer='glorot_uniform',
            trainable=True)
        super().build(input_shape)
 
    def call(self, x):
        seq_len = tf.shape(x)[1]  
        positions = tf.range(start=0,  limit=seq_len, delta=1)
        positions = tf.cast(positions,  tf.int32)  
        return x + tf.nn.embedding_lookup(self.pos_emb,  positions)
 
class MultiHeadSelfAttention(tf.keras.layers.Layer):  
    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim  = embed_dim 
        self.num_heads  = num_heads 
        self.head_dim  = embed_dim // num_heads 
 
        assert self.head_dim  * num_heads == embed_dim 
 
    def build(self, input_shape):
        self.query  = tf.keras.layers.Dense(self.embed_dim)  
        self.key  = tf.keras.layers.Dense(self.embed_dim)  
        self.value  = tf.keras.layers.Dense(self.embed_dim)  
        self.combine  = tf.keras.layers.Dense(self.embed_dim)  
        super().build(input_shape)
 
    def call(self, x):
        batch_size = tf.shape(x)[0]  
 
        q = self.query(x)  
        k = self.key(x)  
        v = self.value(x)  
 
        q = tf.reshape(q,  (batch_size, -1, self.num_heads,  self.head_dim))  
        q = tf.transpose(q,  [0, 2, 1, 3])
 
        k = tf.reshape(k,  (batch_size, -1, self.num_heads,  self.head_dim))  
        k = tf.transpose(k,  [0, 2, 1, 3])
 
        v = tf.reshape(v,  (batch_size, -1, self.num_heads,  self.head_dim))  
        v = tf.transpose(v,  [0, 2, 1, 3])
 
        matmul_qk = tf.matmul(q,  k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1],  tf.float32)  
        scaled_attention = tf.nn.softmax(matmul_qk  / tf.math.sqrt(dk),  axis=-1)
        output = tf.matmul(scaled_attention,  v)
 
        output = tf.transpose(output,  [0, 2, 1, 3])
        output = tf.reshape(output,  (batch_size, -1, self.embed_dim))  
        return self.combine(output)  
 
class TransformerBlock(tf.keras.layers.Layer):  
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim  = embed_dim 
        self.num_heads  = num_heads 
        self.ff_dim  = ff_dim 
        self.rate  = rate 
 
        self.att  = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn  = tf.keras.Sequential([  
            tf.keras.layers.Dense(ff_dim,  activation='swish'),
            tf.keras.layers.Dense(embed_dim)  
        ])
        self.layernorm1  = tf.keras.layers.LayerNormalization(epsilon=1e-6)  
        self.layernorm2  = tf.keras.layers.LayerNormalization(epsilon=1e-6)  
        self.dropout1  = tf.keras.layers.Dropout(rate)  
        self.dropout2  = tf.keras.layers.Dropout(rate)  
 
    def call(self, inputs, training=True):
        attn_output = self.att(inputs)  
        attn_output = self.dropout1(attn_output,  training=training)
        out1 = self.layernorm1(inputs  + attn_output)
 
        ffn_output = self.ffn(out1)  
        ffn_output = self.dropout2(ffn_output,  training=training)
        return self.layernorm2(out1  + ffn_output)
 
class PTMVisualizer:
    def __init__(self, config):
        self.config  = config 
        self.model  = None 
        self._configure_gpu()  # 新增GPU配置
        self.load_model()   


    def _configure_gpu(self):
        """初始化GPU配置"""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # 设置GPU内存按需增长
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"已启用 {len(gpus)} 块GPU的内存动态增长")
            except RuntimeError as e:
                # 如果已经初始化则忽略错误
                print("GPU配置警告:", str(e))
        else:
            print("警告: 未检测到GPU设备，将使用CPU")
    def load_model(self):
        """加载训练好的模型"""
        model_path = os.path.join(   
            self.config.model_dir,   
            f"{self.config.dataset_name}_{self.config.epochs}_{self.config.patience}_{self.config.batch_size}_fold_2_best_model.h5"   
        )
        
        custom_objects = {
            'PositionEmbedding': PositionEmbedding,
            'MultiHeadSelfAttention': MultiHeadSelfAttention,
            'TransformerBlock': TransformerBlock,
            'focal_loss': self.focal_loss,   
            'AdamW': AdamW 
        }
        
        with tf.keras.utils.custom_object_scope(custom_objects):   
            try:
                self.model  = load_model(model_path)
                print(f"成功加载模型: {model_path}")
            except Exception as e:
                print(f"加载模型失败: {str(e)}")
                self.model  = None 
        
    # def focal_loss(self, y_true, y_pred):
    #     """Focal loss函数"""
    #     alpha = 0.25 
    #     gamma = 2.0 
    #     y_pred = K.clip(y_pred,  K.epsilon(),  1.0 - K.epsilon())  
    #     cross_entropy = -y_true * K.log(y_pred)  
    #     loss = alpha * K.pow(1  - y_pred, gamma) * cross_entropy 
    #     return K.mean(K.sum(loss,  axis=1))
    def focal_loss(self, y_true, y_pred):
        fl = SigmoidFocalCrossEntropy(alpha=0.25, gamma=2.0)
        return fl(y_true, y_pred)
        
    def analyze_feature_importance(self, X_test, feature_names, y_test, n_repeats=5):
        """优化后的GPU加速特征重要性分析"""
        if self.model is None:
            print("模型未加载，无法分析特征重要性")
            return None
            
        print("\n=== GPU加速的特征重要性分析 ===")
        
        # 转换数据为TensorFlow张量（自动分配到GPU）
        X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y_test_np = y_test if len(y_test.shape) == 1 else np.argmax(y_test, axis=1)
        
        # 预计算模型输入结构
        total_seq_len = sum(self.config.window_sizes)
        
        @tf.function
        def model_predict(X):
            """GPU加速的模型预测函数"""
            seq_features = X[:, :total_seq_len]
            other_features = X[:, total_seq_len:]
            
            inputs = []
            start = 0
            for win_size in self.config.window_sizes:
                end = start + win_size
                inputs.append(tf.cast(seq_features[:, start:end], tf.int32))
                start = end
            
            struct_features = other_features[:, :-self.config.esm_dim]
            esm_features = other_features[:, -self.config.esm_dim:]
            
            inputs.append(struct_features)
            inputs.append(esm_features)
            
            return self.model(inputs, training=False)[:, 1]
        
        # 基准分数
        baseline_pred = model_predict(X_test_tf)
        baseline_score = roc_auc_score(y_test_np, baseline_pred.numpy())
        
        # 并行计算特征重要性
        importance_scores = []
        std_scores = []
        
        for i in tqdm(range(X_test.shape[1]),  desc="Processing features"):
            perturbed_scores = []
            for _ in range(n_repeats):
                # 创建扰动副本（确保在GPU上）
                X_perturbed = tf.identity(X_test_tf) 
                
                # 生成随机索引（GPU加速）
                indices = tf.random.shuffle(tf.range(tf.shape(X_perturbed)[0])) 
                
                # 关键修复：正确处理二维更新 
                update_indices = tf.stack([ 
                    tf.range(tf.shape(X_perturbed)[0]),   # 行索引 
                    tf.fill([tf.shape(X_perturbed)[0]],  i)  # 列索引 
                ], axis=1)
                
                # 获取待更新列并打乱（保持二维）
                update_values = tf.gather(X_perturbed[:,  i:i+1], indices)
                
                # 执行散射更新 
                X_perturbed = tf.tensor_scatter_nd_update( 
                    X_perturbed,
                    update_indices,
                    tf.squeeze(update_values,  axis=1)  # 确保更新值形状为[N]
                )
                
                # GPU预测
                perturbed_pred = model_predict(X_perturbed)
                score = roc_auc_score(y_test_np, perturbed_pred.numpy())
                perturbed_scores.append(score)
            
            importance_scores.append(baseline_score - np.mean(perturbed_scores))
            std_scores.append(np.std(perturbed_scores))
        
        # 生成结果
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': importance_scores,
            'importance_std': std_scores
        }).sort_values('importance_mean', ascending=False)
        
        # 可视化
        # 修改后（正确版本）：
        top_features = importance_df.head(20) 
        plt.figure(figsize=(12,  8))
        bar_plot = sns.barplot( 
            x='importance_mean',
            y='feature',
            data=top_features,
            ci=None  # 禁用内置误差计算 
        )
         
        # 手动添加误差线（确保维度匹配）
        xerr_values = top_features['importance_std'].values 
        for i, (_, row) in enumerate(top_features.iterrows()): 
            plt.errorbar( 
                x=row['importance_mean'],
                y=i,
                xerr=row['importance_std'],
                fmt='none',
                color='black',
                capsize=3 
            )
         
        plt.xlabel('Feature  Importance Score')
        plt.ylabel('Feature  Name')
        plt.title(f'Top  20 Features for {self.config.ptm_type}  Prediction')
        plt.grid(axis='x',  linestyle='--', alpha=0.6)
        plt.tight_layout() 
        
        output_path = os.path.join(
            self.config.result_dir,
            f"{self.config.ptm_type}_feature_importance_gpu2.png"
        )
        plt.savefig(output_path)
        plt.close()
        print(f"特征重要性图已保存至: {output_path}")
        
        return importance_df
    

    
    def _predict_with_model(self, X):
        """辅助函数：使用模型进行预测"""
        total_seq_len = sum(self.config.window_sizes)  
        
        seq_features = X[:, :total_seq_len]
        other_features = X[:, total_seq_len:]
        
        inputs = []
        start = 0 
        for win_size in self.config.window_sizes:  
            end = start + win_size 
            inputs.append(seq_features[:,  start:end])
            start = end 
        
        struct_features = other_features[:, :-self.config.esm_dim]  
        esm_features = other_features[:, -self.config.esm_dim:]  
        
        inputs.append(struct_features)  
        inputs.append(esm_features)  
        
        return self.model.predict(inputs,  verbose=0)
    
    def visualize_attention(self, sequences, positions):
        """可视化注意力权重"""
        print("\n=== 注意力权重可视化 ===")
        
        win_size = 52 
        half_win = (win_size - 1) // 2 
        
        seq_windows = []
        for seq, pos in zip(sequences, positions):
            center = seq[pos-1]
            left = seq[max(0, pos-1-half_win):pos-1]
            right = seq[pos:min(len(seq), pos+half_win)]
            
            left_pad = max(0, half_win - len(left))
            right_pad = max(0, half_win - len(right))
            
            window = self.config.empty_aa  * left_pad + left + center + right + self.config.empty_aa  * right_pad 
            seq_windows.append(window)  
        
        letter_dict = {
            "A":0, "C":1, "D":2, "E":3, "F":4, "G":5, "H":6, "I":7, "K":8, "L":9,
            "M":10, "N":11, "P":12, "Q":13, "R":14, "S":15, "T":16, "V":17, "W":18, "Y":19,
            "*":20 
        }
        
        encoded = np.zeros((len(seq_windows),  win_size), dtype=int)
        for i, seq in enumerate(seq_windows):
            for j, aa in enumerate(seq):
                encoded[i, j] = letter_dict.get(aa,  20)
        
        seq_input = Input(shape=(win_size,))
        x = Embedding(self.config.max_features,  self.config.embed_dim)(seq_input)  
        x = PositionEmbedding(max_len=win_size, embed_dim=self.config.embed_dim)(x)  
        x = TransformerBlock(self.config.embed_dim,  self.config.num_heads,  self.config.ff_dim)(x)  
        
        attention_layer = MultiHeadSelfAttention(self.config.embed_dim,  self.config.num_heads)  
        attn_output = attention_layer(x)
        
        attention_model = Model(inputs=seq_input, outputs=attn_output)
        
        attn_weights = attention_model.predict(encoded)  
        
        avg_attn = np.mean(attn_weights,  axis=1)
        
        n_examples = min(20, len(sequences))
        plt.figure(figsize=(15,  3*n_examples))
        
        for i in range(n_examples):
            plt.subplot(n_examples,  1, i+1)
            sns.heatmap(  
                avg_attn[i:i+1],
                cmap="YlOrRd",
                xticklabels=list(seq_windows[i]),
                yticklabels=False 
            )
            plt.title(f"Sequence:  {seq_windows[i]}\nPosition: {half_win+1} (Center)")
        
        plt.tight_layout()  
        
        output_path = os.path.join(  
            self.config.result_dir,  
            f"{self.config.ptm_type}_attention_weights6.png"  
        )
        plt.savefig(output_path)  
        plt.close()  
        print(f"注意力权重图已保存至: {output_path}")
 
    def _create_sequence_logo(self, sequences, title):
        """创建序列logo图"""
        if not sequences:
            print(f"警告: 没有可用于创建 {title} logo的序列")
            return 
        
        output_path = os.path.join(  
            self.config.result_dir,  
            f"{title}_logo6.png"  
        )
        
        try:
            from logomaker import alignment_to_matrix 
            df = alignment_to_matrix(sequences)
            
            plt.figure(figsize=(20,  3))
            logo = logomaker.Logo(df, color_scheme='chemistry')
            plt.title(title)  
            plt.savefig(output_path)  
            plt.close()  
            print(f"序列logo图已保存至: {output_path}")
            
        except Exception as e:
            print(f"创建序列logo时出错: {str(e)}")
            print("尝试使用替代方法创建logo...")
            self._alternative_create_logo(sequences, title, output_path)
    
    def _alternative_create_logo(self, sequences, title, output_path):
        """替代方法创建序列logo"""
        try:
            aa_list = sorted(set(aa for seq in sequences for aa in seq))
            if not aa_list:
                print("错误: 没有有效的氨基酸序列")
                return 
                
            freq_matrix = []
            max_len = max(len(seq) for seq in sequences)
            
            for i in range(max_len):
                pos_counts = {}
                total = 0 
                for seq in sequences:
                    if i < len(seq):
                        aa = seq[i]
                        pos_counts[aa] = pos_counts.get(aa,  0) + 1 
                        total += 1 
                
                pos_freq = {aa: count/total for aa, count in pos_counts.items()} 
                freq_matrix.append(pos_freq) 
            
            import pandas as pd 
            df = pd.DataFrame(freq_matrix).fillna(0)
            
            plt.figure(figsize=(20,  3))
            logo = logomaker.Logo(df, color_scheme='chemistry')
            plt.title(title) 
            plt.savefig(output_path) 
            plt.close() 
            print(f"使用替代方法创建的序列logo图已保存至: {output_path}")
            
        except Exception as e:
            print(f"替代方法也失败: {str(e)}")
            print("无法创建序列logo图")
    
    def analyze_motifs(self, sequences, positions, labels, win_size=15):
        """分析修饰位点周围的序列模式"""
        print("\n=== 序列模式分析 ===")
        
        half_win = win_size // 2
        motifs = {'positive': [], 'negative': []}
        
        # 提取阳性样本和阴性样本的序列窗口
        for seq, pos, label in zip(sequences, positions, labels):
            if not isinstance(seq, str) or len(seq) == 0:
                continue  # 跳过无效序列
                
            start = max(0, pos-1-half_win)
            end = min(len(seq), pos+half_win)
            window = seq[start:end]
            
            # 填充不足的窗口 
            if len(window) < win_size:
                left_pad = max(0, half_win - (pos-1))
                right_pad = max(0, half_win - (len(seq) - pos))
                window = 'X'*left_pad + window + 'X'*right_pad
            
            # 确保窗口长度正确
            if len(window) != win_size:
                continue
                
            if label == 1:
                motifs['positive'].append(window)
            else:
                motifs['negative'].append(window)
        
        # 检查是否有足够样本
        min_samples = 5
        if len(motifs['positive']) < min_samples:
            print(f"警告: 阳性样本数量不足({len(motifs['positive'])}), 至少需要{min_samples}个样本")
        else:
            self._create_sequence_logo(motifs['positive'], f"{self.config.ptm_type}_positive_motif")  
        
        if len(motifs['negative']) < min_samples:
            print(f"警告: 阴性样本数量不足({len(motifs['negative'])}), 至少需要{min_samples}个样本")
        else:
            self._create_sequence_logo(motifs['negative'], f"{self.config.ptm_type}_negative_motif")  
    
        # 保存motif序列 
        with open(os.path.join(self.config.result_dir, f"{self.config.ptm_type}_motif_sequences6.txt"), 'w') as f:
            f.write(f"=== Positive Samples ({len(motifs['positive'])} sequences) ===\n")
            f.write("\n".join(motifs['positive']) + "\n\n")
            f.write(f"=== Negative Samples ({len(motifs['negative'])} sequences) ===\n")
            f.write("\n".join(motifs['negative']) + "\n")



 
    def visualize_decision_boundary(self, X_test, y_test):
        """可视化模型的决策边界"""
        print("\n=== 决策边界可视化 ===")
        
        # 获取模型的预测概率
        y_pred = self.model.predict(X_test)[:,  1]
        
        # 绘制概率分布
        plt.figure(figsize=(10,  6))
        sns.histplot( 
            data=pd.DataFrame({
                'Probability': y_pred,
                'Class': ['Positive' if x == 1 else 'Negative' for x in y_test]
            }),
            x='Probability',
            hue='Class',
            bins=50,
            kde=True,
            element='step',
            common_norm=False
        )
        plt.title('Predicted  Probability Distribution by True Class')
        plt.xlabel('Predicted  Probability')
        plt.ylabel('Count') 
        
        # 保存结果 
        output_path = os.path.join( 
            self.config.result_dir, 
            f"{self.config.ptm_type}_decision_boundary6.png" 
        )
        plt.savefig(output_path) 
        plt.close() 
        print(f"决策边界图已保存至: {output_path}")
 
def load_data_for_visualization(config):
    """加载用于可视化的数据"""
    # 加载原始数据集 
    data_path = os.path.join(config.data_dir,  f"{config.dataset_name}.csv") 
    df = pd.read_csv(data_path) 
    
    # 根据当前PTM类型筛选数据
    if config.ptm_type  == 'phosphorylation':
        df = df[df['ptm'].str.contains('Phosphorylation',  case=False)]
        df = df[df['mod_aa'].isin(['S', 'T'])]
    elif config.ptm_type  == 'acetylation':
        df = df[df['ptm'].str.contains('Acetylation',  case=False)]
        df = df[df['mod_aa'] == 'K']
    elif config.ptm_type  == 'ubiquitination':
        df = df[df['ptm'].str.contains('Ubiquitination',  case=False)]
        df = df[df['mod_aa'] == 'K']
    
    # 验证位置是否有效并过滤无效样本
    valid_mask = df.apply(lambda  x: 1 <= x['pos'] <= len(x['trunc_seq']), axis=1)
    df = df[valid_mask]
    
    # 提取必要字段 
    sequences = df['trunc_seq'].values
    positions = df['pos'].values 
    labels = df['label'].values
    protein_ids = df['entry'].values
    
    # 处理结构特征 
    df['phi'] = df['phi'].apply(lambda x: np.array(eval(x))) 
    df['psi'] = df['psi'].apply(lambda x: np.array(eval(x))) 
    
    # 提取中心位置特征
    half_window = (max(config.window_sizes)  - 1) // 2
    df['phi_center'] = df['phi'].apply(lambda x: x[half_window] if len(x) > half_window else 0.0)
    df['psi_center'] = df['psi'].apply(lambda x: x[half_window] if len(x) > half_window else 0.0)
    
    # 构建结构特征矩阵 
    struct_features = np.column_stack([ 
        df['sasa'].values,
        df['phi_center'].values,
        df['psi_center'].values,
        pd.get_dummies(df['secstruct']).values, 
        df['local_plDDT'].values,
        df['avg_plDDT'].values 
    ])
    
    # 加载ESM特征 
    data_loader = PTMDataLoader(config)
    esm_dict = data_loader.load_esm_features() 
    
    esm_features = []
    for protein_id, pos in zip(protein_ids, positions):
        lookup_key = f"{protein_id}_{pos}"
        if esm_dict is not None and lookup_key in esm_dict:
            esm_features.append(esm_dict[lookup_key]['features']) 
        else:
            esm_features.append(np.zeros(config.esm_dim)) 
    
    esm_features = np.array(esm_features) 
    
    return sequences, positions, labels, protein_ids, struct_features, esm_features 
 
# 主程序
if __name__ == "__main__":
    # 选择要分析的PTM类型
    ptm_type = 'phosphorylation'  # 可以改为'acetylation'或'ubiquitination'
 
    print(f"\n{'='*50}")
    print(f"正在分析: {ptm_type.upper()}  PTM模型")
    print(f"{'='*50}")
 
    # 创建配置和可视化器 
    config = Config(ptm_type=ptm_type)
    visualizer = PTMVisualizer(config)
 
    # 加载数据 - 现在会过滤无效位置
    sequences, positions, labels, protein_ids, struct_features, esm_features = load_data_for_visualization(config)
 
    # 准备多尺度序列特征
    X_seq = []
    data_loader = PTMDataLoader(config)
    for win_size in config.window_sizes: 
        seq_windows = data_loader.prepare_samples(sequences,  positions, win_size)
        X_seq.append(data_loader.encode_sequences(seq_windows)) 
 
    # 检查所有特征的样本数量是否一致 
    num_samples = len(labels)
    assert all(x.shape[0]  == num_samples for x in X_seq), "序列特征样本数量不一致"
    assert struct_features.shape[0]  == num_samples, "结构特征样本数量不一致"
    assert esm_features.shape[0]  == num_samples, "ESM特征样本数量不一致"
 
    # 合并所有特征用于重要性分析
    X_test_combined = np.concatenate( 
        [x.reshape(num_samples, -1) for x in X_seq] +
        [struct_features, esm_features],
        axis=1
    )
 
    # 准备特征名称 
    struct_feature_names = [
        'sasa', 'phi_center', 'psi_center',
        'secstruct_E', 'secstruct_H', 'secstruct_L',
        'local_plDDT', 'avg_plDDT'
    ]
 
    # 为每个窗口大小添加序列特征名称
    seq_feature_names = []
    for i, win_size in enumerate(config.window_sizes): 
        seq_feature_names += [f"win{i}_pos{j}" for j in range(win_size)]
 
    feature_names = seq_feature_names + struct_feature_names + [f"ESM_{i}" for i in range(config.esm_dim)] 
 
    # 执行各种分析
    print("\n1. 特征重要性分析...")
    visualizer.analyze_feature_importance(X_test_combined,  feature_names, labels)
 
    print("\n2. 注意力权重可视化...")
    # 选择部分样本以避免内存问题
    sample_idx = np.random.choice(len(sequences),  size=min(100, len(sequences)), replace=False)
    visualizer.visualize_attention( 
        np.array(sequences)[sample_idx], 
        np.array(positions)[sample_idx] 
    )
 
    print("\n3. 序列模式分析...")
    visualizer.analyze_motifs(sequences,  positions, labels)
 

 
    print("\n5. 决策边界分析...")
    # 使用部分样本
    sample_idx = np.random.choice(len(X_test_combined),  size=min(10000, len(X_test_combined)), replace=False)
    visualizer.visualize_decision_boundary( 
        [x[sample_idx] for x in X_seq] + [struct_features[sample_idx], esm_features[sample_idx]],
        labels[sample_idx]
    )