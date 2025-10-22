import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (Layer, Dense, Embedding, Dropout, LayerNormalization,
                                    Conv1D, GlobalAveragePooling1D, Input, Add,
                                    Concatenate, BatchNormalization, Activation,
                                    Flatten, RepeatVector, Permute, Multiply, Lambda,
                                    SpatialDropout1D)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (LearningRateScheduler, EarlyStopping,
                                       ModelCheckpoint, CSVLogger, TensorBoard)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score, matthews_corrcoef,
                            roc_auc_score, precision_score, recall_score,
                            precision_recall_curve, auc, average_precision_score,
                            classification_report)
from tensorflow.keras import backend as K
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import json
import torch
import pyrosetta
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import re

# --- Dynamic Path Determination ---
# Get the directory where this script (predict.py) is located
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Determine the DeepPTMPred project root by going up two levels from the script's directory
# If predict.py is at DeepPTMPred/pred/train_PTM/, then current_script_dir is DeepPTMPred/pred/train_PTM/
# Going up two levels gets to DeepPTMPred/
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

print(f"DeepPTMPred project root determined as: {project_root}")
# --- End Dynamic Path Determination ---


def extract_protein_id_from_pdb_path(pdb_path):
    """
    从 AlphaFold PDB 文件路径中提取 UniProt ID
    支持格式：AF-P12345-F1-model_v4.pdb 或 AF_Q99999_F1_model_v3.pdb
    """
    filename = os.path.basename(pdb_path)

    # 匹配模式1: AF-{id}-F1-model_vX.pdb
    match = re.search(r'AF-([A-Za-z0-9]+)-F\d+-model_v\d+', filename)
    if match:
        return match.group(1)

    # 匹配模式2: AF_{id}_F1_model_vX.pdb
    match = re.search(r'AF_([A-Za-z0-9]+)_F\d+_model_v\d+', filename)
    if match:
        return match.group(1)

    # 其他常见格式（如直接是 P12345.pdb）
    match = re.search(r'^([A-Za-z0-9]{5,6})\.pdb$', filename)
    if match:
        return match.group(1)

    raise ValueError(f"无法从文件名 '{filename}' 中提取 protein_id")

# 标准氨基酸三字母到单字母的映射
one_letter = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'SEC': 'U', 'PYL': 'O', 'ASX': 'B', 'GLX': 'Z', 'XAA': 'X', 'XLE': 'J'
}

def extract_sequence_from_pdb(pdb_path, chain_id=None):
    """
    从 PDB 文件中提取氨基酸序列
    :param pdb_path: PDB 文件路径
    :param chain_id: 指定链 ID（如 'A'），若为 None 则合并所有链
    :return: 氨基酸序列（字符串）
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    sequences = {}

    for model in structure:
        for chain in model:
            chain_id_actual = chain.id # Renamed to avoid conflict with function parameter
            seq = []
            for residue in chain:
                if residue.get_id()[0] != " ":
                    continue
                resname = residue.get_resname()
                try:
                    seq.append(one_letter[resname])
                except KeyError:
                    # 对于未知残基，用 'X' 表示
                    seq.append('X')
            sequences[chain_id_actual] = ''.join(seq)

    # 如果只想要一条链
    if chain_id is not None:
        return sequences.get(chain_id, "")

    # 否则返回所有链的序列（用 '/' 分隔）
    return '/'.join([f"{k}:{v}" for k, v in sequences.items()])

# ==================== 配置类 (优化版) ====================
class PredictConfig:
    def __init__(self, ptm_type='phosphorylation', project_root=None): # Added project_root parameter

        # PTM类型和目标氨基酸映射
        self.ptm_aa_map = {
            'phosphorylation': ['S', 'T'],
            'ubiquitination': ['K'],
            'acetylation': ['K'],
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

        # 设置PTM类型和目标氨基酸
        self.ptm_type = ptm_type
        self.target_aa = self.ptm_aa_map.get(ptm_type, [])

        if not self.target_aa:
            raise ValueError(f"不支持的PTM类型: {ptm_type}")

        # 路径配置 - 现在使用传递进来的 project_root
        if project_root is None:
            raise ValueError("project_root must be provided to PredictConfig")

        self.root_dir = project_root
        # Use os.path.join for platform-independent path construction
        self.esm_dir = os.path.join(self.root_dir, "pred", "train_PTM")
        self.custom_esm_dir = os.path.join(self.root_dir, "pred", "custom_esm")

        # 模型路径 - 根据PTM类型动态生成
        self.model_path = os.path.join(
            self.esm_dir,
            "model",
            f"models_{ptm_type}_esm2",
            "ptm_data_210_39_64_best_model.h5"
        )

        self.esm_dim = 1280
        self.struct_features = ['sasa', 'phi', 'psi', 'secstruct', 'local_plDDT', 'avg_plDDT']
        self.struct_feature_dim = 8
        self.embed_dim = 64
        self.num_heads = 4
        self.ff_dim = 256
        self.window_sizes = [21, 33, 51]


class PyRosettaCalculator:
    def __init__(self, pdb_path=None):
        # 调试信息：打印PyRosetta初始化状态
        print("\n=== PyRosetta Initialization ===")
        try:
            # 初始化 PyRosetta（忽略未知残基、静默模式）
            pyrosetta.init("-ignore_unrecognized_res -mute all -ignore_zero_occupancy false")
            print("Successfully initialize PyRosetta")
        except Exception as e:
            print(f"❌ PyRosetta initialazation fail: {str(e)}")
            raise

        self.pose = None
        self.plDDT_values = []
        self.res_sasa = {}

        if pdb_path:
            # 调试信息：验证文件可读性
            abs_path = os.path.abspath(pdb_path)
            print(f"\n=== PDB File Validation ===")
            print(f"file path: {abs_path}")
            print(f"file exist: {os.path.exists(pdb_path)}")
            if not os.path.exists(pdb_path):
                raise FileNotFoundError(f"PDB file does not exist: {pdb_path}")

            print(f"file size: {os.path.getsize(pdb_path)} bytes")


            try:
                # 加载PDB
                print("\n=== PyRosetta load PDB ===")
                self.pose = pyrosetta.pose_from_pdb(pdb_path)
                print(f"Loaded successfully! Total number of residues: {self.pose.total_residue()}")
                # residue_1 = self.pose.residue(1) # Commented out unused variable
                # pdb_id = self.pose.pdb_info().pose2pdb(1) # Commented out unused variable
                # print(f"第一个残基: {residue_1.name()} (编号 {pdb_id})")


                #=== 计算SASA：强制初始化 + 显式设置参数 ===
                # === 计算SASA ===
                print("\n=== Calculate SASA ===")
                try:
                    from pyrosetta.rosetta.core.scoring import calc_per_atom_sasa
                    from pyrosetta.rosetta.core.id import AtomID_Map_double_t
                    from pyrosetta.rosetta.utility import vector1_double

                    # 创建正确的参数类型
                    atom_sasa = AtomID_Map_double_t()
                    residue_sasa = vector1_double()

                    # 初始化 atom_sasa（必须与 pose 的原子结构匹配）
                    atom_sasa.resize(self.pose.size())
                    for i in range(1, self.pose.total_residue() + 1):
                        atom_sasa[i].resize(self.pose.residue(i).natoms())

                    # 计算 SASA
                    calc_per_atom_sasa(
                        self.pose,
                        atom_sasa,
                        residue_sasa,
                        1.4  # 探针半径
                    )

                    # 提取残基级 SASA
                    for i in range(1, self.pose.total_residue() + 1):
                        total_sasa = residue_sasa[i]
                        # 简单分配：主链 SASA 约为总 SASA 的 30%，侧链 70%
                        bb_sasa = total_sasa * 0.3
                        sc_sasa = total_sasa * 0.7
                        self.res_sasa[i] = (total_sasa, bb_sasa, sc_sasa)

                    # 打印前5个残基的 SASA 值
                    # sample_res = min(5, self.pose.total_residue()) # Commented out unused variable
                    print(f"SASA calculation successful!")
                    # for i in range(1, sample_res + 1):
                    #     t, p, n = self.res_sasa[i]
                    #     print(f"  Res {i}: Total={t:.1f}Å² | Polar(BB)={p:.1f}Å² | Nonpolar(SC)={n:.1f}Å²")

                except Exception as e:
                    print(f"SASA calculation fails: {str(e)}")
                    self.res_sasa = {i: (0.0, 0.0, 0.0) for i in range(1, self.pose.total_residue() + 1)}



                # === 提取plDDT（从B-factor）===
                print("\n=== Extracting plDDT (B-factor) ===")
                parser = PDBParser()
                structure = parser.get_structure("protein", pdb_path)
                b_factors = [atom.get_bfactor() for atom in structure.get_atoms()]
                self.plDDT_values = b_factors
                avg_plddt = np.mean(b_factors)
                print(f"Extracted {len(b_factors)} B-factors（plDDT）")
                print(f"Average plDDT: {avg_plddt:.2f}")

            except Exception as e:
                print(f"\nPyRosetta load PDB failed: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
        else:
            print("PDB path not provided")
    def calculate_features(self, residue_number):
        """带完整错误处理的特征计算（8维输出）"""
        try:
            if not self.pose:
                raise ValueError("PyRosetta未加载PDB结构")
            if residue_number < 1 or residue_number > self.pose.total_residue():
                raise ValueError(f"残基编号 {residue_number} 超出范围 [1, {self.pose.total_residue()}]")

            # print(f"\n🔍 计算残基 {residue_number} 特征...")

            # 1. SASA：只保留 total_sasa
            total_sasa, _, _ = self.res_sasa.get(residue_number, (0.0, 0.0, 0.0))
            # print(f"   SASA (total): {total_sasa:.2f} Å²")

            # 2. 二面角
            phi = self.pose.phi(residue_number)
            psi = self.pose.psi(residue_number)
            # print(f"   phi={phi:.1f}, psi={psi:.1f}")

            # 3. 二级结构（one-hot 编码：H=helix, E=sheet, L=loop）
            ss = self.pose.secstruct(residue_number)
            # 顺序：H, E, L（对应 one-hot）
            ss_onehot = [1, 0, 0] if ss == 'H' else ([0, 1, 0] if ss == 'E' else [0, 0, 1])
            # print(f"   二级结构: {ss} -> one-hot: {ss_onehot}")

            # 4. plDDT
            local_plddt = (
                self.plDDT_values[residue_number - 1]
                if self.plDDT_values and residue_number <= len(self.plDDT_values)
                else 85.0
            )
            avg_plddt = np.mean(self.plDDT_values) if self.plDDT_values else 85.0
            # print(f"   local_plDDT={local_plddt:.1f}, avg_plDDT={avg_plddt:.1f}")


            # 特征顺序：sasa, phi, psi, H, E, L, local_plDDT, avg_plDDT
            features = [
                total_sasa,           # sasa (1)
                phi, psi,             # phi, psi (2)
                ss_onehot[0],         # secstruct: H (1)
                ss_onehot[1],         # secstruct: E (1)
                ss_onehot[2],         # secstruct: L (1)
                local_plddt,          # local_plDDT (1)
                avg_plddt             # avg_plDDT (1)
            ]
            return np.array(features)
        except Exception as e:
            print(f" 残基 {residue_number} 特征计算失败: {str(e)}")
            return np.array([0.0, 0.0, 0.0, 0, 0, 1, 85.0, 85.0])  # 默认 sasa=0, phi=0, psi=0, 结构=L, plDDT=85
# ==================== 自定义层定义 (需与训练时一致) ====================
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
# ==================== 数据加载与处理 (优化版) ====================
class PTMPredictDataLoader:
    def __init__(self, config):
        self.config = config
        self.rosetta_calc = None

    def init_pyrosetta(self, pdb_path):
        """延迟初始化PyRosetta"""
        if self.rosetta_calc is None:
            self.rosetta_calc = PyRosettaCalculator(pdb_path)

    def load_custom_esm(self, protein_id):
        """加载单个蛋白质的完整ESM特征"""
        esm_path = os.path.join(self.config.custom_esm_dir, f"{protein_id}_full_esm.npz")
        if os.path.exists(esm_path):
            try:
                data = np.load(esm_path)
                print(f"成功加载ESM特征文件: {esm_path}")
                return data['features']  # 形状: [seq_len, esm_dim]
            except Exception as e:
                print(f"加载ESM特征文件失败: {esm_path}, 错误: {str(e)}")
                return None
        else:
            print(f"ESM特征文件不存在: {esm_path}")
            return None

    def prepare_protein_data(self, protein_id, protein_sequence, positions=None, pdb_path=None):
        """优化后的数据准备方法"""
        # 确定预测位点
        if positions is None:
            positions = [i+1 for i, aa in enumerate(protein_sequence)
                        if aa in self.config.target_aa]

        # 初始化 PyRosetta（延迟）
        if pdb_path:
            self.init_pyrosetta(pdb_path)  # 这会创建 self.rosetta_calc

        # 加载ESM特征
        esm_features = self._prepare_esm_features(protein_id, protein_sequence, positions)

        # 生成结构特征
        struct_features = self._generate_structural_features(positions)  # 内部使用 self.rosetta_calc

        # 准备序列窗口
        X_seq = []
        for win_size in self.config.window_sizes:
            seq_windows = self._get_sequence_windows(protein_sequence, positions, win_size)
            X_seq.append(self._encode_sequences(seq_windows))

        return X_seq, struct_features, esm_features, positions

    def _prepare_esm_features(self, protein_id, protein_sequence, positions):
        """ESM特征处理优化"""
        custom_esm = self.load_custom_esm(protein_id)
        esm_features = []

        for pos in positions:
            if custom_esm is not None and pos-1 < len(custom_esm):
                # 提取目标位置ESM特征并增强关键位点
                feat = custom_esm[pos-1].copy()

                esm_features.append(feat)
            else:
                # 回退到零填充
                esm_features.append(np.zeros(self.config.esm_dim))
        return np.array(esm_features)

    def _generate_structural_features(self, positions):
        """
        生成结构特征（SASA, phi, psi, 二级结构）
        """
        features = np.zeros((len(positions), self.config.struct_feature_dim))

        if self.rosetta_calc is not None:
            for i, pos in enumerate(positions):
                try:
                    # 调用 PyRosetta 计算单个残基特征
                    feat = self.rosetta_calc.calculate_features(pos)
                    features[i] = feat
                except Exception as e:
                    print(f"警告：残基 {pos} 结构特征计算失败: {str(e)}")
                    features[i] = self._get_default_features()
        else:
            # 如果没有结构，返回默认值
            features = np.array([self._get_default_features() for _ in positions])

        return features
    
    def _get_default_features(self):
        """默认特征值（计算失败时使用）"""
        return np.array([0.5, 0.0, 0.0, 0, 1, 0,85.0,85.0])  # [sasa, phi, psi, H, C, E]


    def _get_sequence_windows(self, sequence, positions, window_size):
        """获取序列窗口"""
        half_len = (window_size - 1) // 2
        windows = []

        for pos in positions:
            center = sequence[pos-1]
            left = sequence[max(0, pos-1-half_len):pos-1]
            right = sequence[pos:min(len(sequence), pos+half_len)]

            left_pad = max(0, half_len - len(left))
            right_pad = max(0, half_len - len(right))

            window = '*' * left_pad + left + center + right + '*' * right_pad
            windows.append(window)

        return windows

    def _encode_sequences(self, sequences):
        """序列编码"""
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

# ==================== 预测器 (优化版) ====================
class PTMPredictor:
    def __init__(self, config):
        self.config = config
        self.data_loader = PTMPredictDataLoader(config)

        # 加载模型 (包含自定义层)
        self.model = load_model(
            config.model_path,
            custom_objects={
                'PositionEmbedding': PositionEmbedding,
                'MultiHeadSelfAttention': MultiHeadSelfAttention,
                'TransformerBlock': TransformerBlock,
                'focal_loss': self.focal_loss
            }
        )

    def focal_loss(self, y_true, y_pred):
        """保持与训练一致的损失函数"""
        fl = tf.keras.losses.SigmoidFocalCrossEntropy(alpha=0.25, gamma=2.0)
        return fl(y_true, y_pred)

    def predict_ptm_sites(self, protein_id, protein_sequence, positions=None, pdb_path=None):
        """
        预测方法 - 所有位点一致对待
        """
        # 初始化PyRosetta（如果有PDB路径）
        self.data_loader.init_pyrosetta(pdb_path)

        # 准备数据
        X_seq, struct_features, esm_features, positions = \
            self.data_loader.prepare_protein_data(protein_id, protein_sequence, positions, pdb_path)

        # 预测概率
        # Ensure that input lists match what the model expects (e.g., if model has 3 sequence inputs + 2 feature inputs)
        # Assuming X_seq is a list of 3 arrays (for 3 window sizes) and model expects 5 inputs total.
        model_inputs = X_seq + [struct_features, esm_features]
        y_pred_proba = self.model.predict(model_inputs)[:, 1]

        # 生成结果DataFrame
        results = []
        for pos, proba in zip(positions, y_pred_proba):
            results.append({
                'protein_id': protein_id,
                'position': pos,
                'residue': protein_sequence[pos-1],
                'probability': proba,
                'prediction': 1 if proba >= 0.5 else 0
            })

        return pd.DataFrame(results)

# ==================== 主程序 ====================
if __name__ == "__main__":

    ptm_type = 'ubiquitination'
    
    # Pass the dynamically determined project_root to PredictConfig
    config = PredictConfig(ptm_type=ptm_type, project_root=project_root)
    predictor = PTMPredictor(config)

    # Construct pdb_path relative to project_root
    pdb_path = os.path.join(project_root, "data", "AF-P31749-F1-model_v4.pdb")
    
    # Ensure the PDB file exists before proceeding
    if not os.path.exists(pdb_path):
        print(f"Error: PDB file not found at {pdb_path}. Please ensure it is in the data folder.")
        exit() # Exit if a critical file is missing

    protein_id = extract_protein_id_from_pdb_path(pdb_path)
    protein_sequence = extract_sequence_from_pdb(pdb_path, chain_id="A")

    # 根据PTM类型找出对应的目标氨基酸位点
    target_aa = config.target_aa
    target_positions = [i+1 for i, aa in enumerate(protein_sequence) if aa in target_aa]

    # 运行预测
    results_df = predictor.predict_ptm_sites(protein_id, protein_sequence, target_positions, pdb_path=pdb_path)

    # 保存结果 - Construct output_path relative to project_root
    output_dir = os.path.join(config.root_dir, "results")
    os.makedirs(output_dir, exist_ok=True) # Ensure results directory exists
    output_path = os.path.join(output_dir, f"{protein_id}_{ptm_type}_predictions.csv")
    results_df.to_csv(output_path, index=False)

    # 打印结果摘要
    print(f"=== {ptm_type} 预测结果摘要 ===")
    print(f"目标氨基酸: {target_aa}")
    print(f"总{''.join(target_aa)}位点数: {len(results_df)}")
    print(f"预测为{ptm_type}的位点数: {len(results_df[results_df['prediction'] == 1])}")
    print(f"\n完整结果已保存至: {output_path}")

    # 显示高概率位点
    high_prob_sites = results_df[results_df['probability'] > 0.7].sort_values('probability', ascending=False)
    if len(high_prob_sites) > 0:
        print(f"\n高概率位点 (概率 > 0.7):")
        for _, row in high_prob_sites.head(10).iterrows():
            print(f"位置 {row['position']} ({row['residue']}): 预测概率 = {row['probability']:.3f}")
