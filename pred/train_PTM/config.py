import os
from typing import Dict, List
class Config:
    def __init__(self, ptm_type='phosphorylation'):   
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

        self.check_esm_validity = True  # 检查ESM特征有效性
        self.filter_invalid_esm = False  # 过滤无效ESM样本
        # 路径配置 
        self.root_dir  = "/root/autodl-tmp/DeepPTMPred"
        self.esm_dir  = "/root/autodl-tmp/DeepPTMPred/pred/train_PTM"
        self.esm2  = "/root/autodl-tmp/DeepPTMPred/pred"
        self.result_dir  = os.path.join(self.esm_dir, "result" , f"results_{self.ptm_type}_esm2_kfold") 
        self.model_dir  = os.path.join(self.esm_dir,  "model"  , f"models_{self.ptm_type}_esm2_kfold") 
        self.data_dir  = os.path.join(self.root_dir,  "data")
        
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
        self.patience  =30
        self.warmup_epochs  = 5 
        
        # 数据参数 
        self.window_sizes  = [21, 33, 51]
        self.max_features  = 21 
        self.empty_aa  = '*'
        self.test_size  = 0.2 
        # K折交叉验证参数
        self.k_folds = 10  # 使用10折交叉验证 
        self.random_state  = 42  # 随机种子
        # ESM配置 
        self.use_esm  = True 
        self.esm_dim  = 1280  # ESM2-t33_650M_UR50D的特征维度 
        
        # 结构特征配置 
        self.struct_features  = ['sasa', 'phi', 'psi', 'secstruct', 'local_plDDT', 'avg_plDDT']
        self.struct_feature_dim  = 8  # 将在初始化时计算 
 
