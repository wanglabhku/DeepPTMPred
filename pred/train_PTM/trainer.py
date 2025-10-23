import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, matthews_corrcoef,
    roc_auc_score, precision_score, recall_score,
    precision_recall_curve, auc, average_precision_score,
    classification_report, roc_curve
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.models import Model, load_model  # 确保包含load_model
from tensorflow.keras.callbacks import (
    LearningRateScheduler, EarlyStopping,
    ModelCheckpoint, CSVLogger, TensorBoard
)
from imblearn.over_sampling import SMOTE
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow_addons.optimizers import AdamW
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from data_loader import PTMDataLoader
from model import create_ptm_model
from model import PositionEmbedding, MultiHeadSelfAttention, TransformerBlock

# ==================== 训练和评估 ====================
class PTMTrainer:
    def __init__(self, config):
        self.config = config
        self.data_loader = PTMDataLoader(config)
        self.model = None
        
    def focal_loss(self, y_true, y_pred):
        
        y_true_labels = K.argmax(y_true, axis=-1)
        N = K.cast(K.shape(y_true)[0], dtype='float32')  
        N_pos = K.cast(K.sum(y_true_labels), dtype='float32')  
        N_neg = N - N_pos  
        
        w_pos = N / (2.0 * N_pos + K.epsilon()) 
        w_neg = N / (2.0 * N_neg + K.epsilon()) 
        
        # 为每个样本分配权重
        weights = w_pos * y_true_labels + w_neg * (1 - y_true_labels)
        # 计算二元交叉熵
        bce = K.binary_crossentropy(y_true, y_pred)
        
        # 应用权重
        weighted_bce = weights * bce
        
        return K.mean(weighted_bce)
    
    def lr_scheduler(self, epoch, lr):
        if epoch < self.config.warmup_epochs:
            return self.config.initial_learning_rate * (epoch + 1) / self.config.warmup_epochs
        return lr * 0.95
    
    def evaluate_model(self, model, X_test, y_test, threshold=0.5):
        y_pred_proba = model.predict(X_test, batch_size=self.config.batch_size*2)
        y_pred = (y_pred_proba[:, 1] > threshold).astype(int)
        y_true = np.argmax(y_test, axis=1)
        y_scores = y_pred_proba[:, 1]
        
        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # 计算FP率和FN率
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        metrics = {
            'Confusion Matrix': [[tn, fp], [fn, tp]],
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1': f1_score(y_true, y_pred),
            'MCC': matthews_corrcoef(y_true, y_pred),
            'AUC': roc_auc_score(y_true, y_scores),
            'AUPR': average_precision_score(y_true, y_scores),
            'FP Rate': fp_rate,
            'FN Rate': fn_rate,
            'Class Report': classification_report(y_true, y_pred, output_dict=True)
        }
        
        # 绘制评估曲线
        self.plot_evaluation_curves(y_true, y_scores, metrics)
        
        return metrics
    
    def plot_evaluation_curves(self, y_true, y_scores, metrics):
        plt.figure(figsize=(15, 5))
        
        # ROC曲线
        plt.subplot(1, 3, 1)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plt.plot(fpr, tpr, label=f'AUC = {metrics["AUC"]:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        
        # PR曲线
        plt.subplot(1, 3, 2)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        plt.plot(recall, precision, label=f'AUPR = {metrics["AUPR"]:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR Curve')
        plt.legend()
        
        # 混淆矩阵
        plt.subplot(1, 3, 3)
        sns.heatmap(metrics['Confusion Matrix'], annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.tight_layout()
        # 修改文件名，包含fold信息
        fold_info = f"_fold_{self.current_fold}" if hasattr(self, 'current_fold') else ""
        plot_path = os.path.join(self.config.result_dir, 
                               f"{self.config.dataset_name}_{self.config.epochs}_{self.config.patience}_{self.config.batch_size}{fold_info}_evaluation_metrics.png")
        plt.savefig(plot_path)
        plt.close()
    
    def optimize_threshold_high_specificity(self, model, X_val, y_val):
        """用于磷酸化的高特异性阈值优化"""
        y_val_proba = model.predict(X_val)[:, 1]
        y_val_labels = np.argmax(y_val, axis=1)
        
        # 强制保持最小特异性
        fpr, tpr, thresholds = roc_curve(y_val_labels, y_val_proba)
        min_specificity = 0.82  # 85%的特异性
        valid_thresholds = thresholds[fpr <= (1 - min_specificity)]
        
        if len(valid_thresholds) > 0:
            best_thresh = valid_thresholds[np.argmax(tpr[fpr <= (1 - min_specificity)])]
        else:
            best_thresh = 0.5  # 回退到默认阈值
        
        return best_thresh
    
    def optimize_threshold_high_specificity2(self, model, X_val, y_val):
        """用于磷酸化的高特异性阈值优化"""
        y_val_proba = model.predict(X_val)[:, 1]
        y_val_labels = np.argmax(y_val, axis=1)
        
        # 强制保持最小特异性
        fpr, tpr, thresholds = roc_curve(y_val_labels, y_val_proba)
        min_specificity = 0.80  # 85%的特异性
        valid_thresholds = thresholds[fpr <= (1 - min_specificity)]
        
        if len(valid_thresholds) > 0:
            best_thresh = valid_thresholds[np.argmax(tpr[fpr <= (1 - min_specificity)])]
        else:
            best_thresh = 0.5  # 回退到默认阈值
        
        return best_thresh

    def optimize_threshold_f1(self, model, X_val, y_val):
        """用于乙酰化和泛素化的F1最大化阈值优化"""
        y_pred_proba = model.predict(X_val) 
        precision, recall, thresholds = precision_recall_curve(
            np.argmax(y_val,  axis=1),
            y_pred_proba[:, 1]
        )
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores) 
        return thresholds[best_idx]
    
    def plot_training_history(self, history, save_path):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(history.history['auc'], label='Train')
        plt.plot(history.history['val_auc'], label='Validation')
        plt.title('Model AUC')
        plt.ylabel('AUC')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def prepare_training_data(self):
        """准备训练数据"""
        # 加载原始数据 
        sequences, positions, labels, protein_ids, struct_features, esm_features = self.data_loader.load_dataset() 
        print(f"结构特征形状: {struct_features.shape}") 
        print(f"ESM特征形状: {esm_features.shape}") 
        
        # 准备多尺度序列特征 
        X_seq = []
        for win_size in self.config.window_sizes: 
            seq_windows = self.data_loader.prepare_samples(sequences,  positions, win_size)
            X_seq.append(self.data_loader.encode_sequences(seq_windows)) 
 
        # 合并所有特征用于K折划分 
        X_all = X_seq + [struct_features, esm_features]
        y_all = to_categorical(labels)
        
        return X_all, y_all

    def handle_class_imbalance(self, X_train, y_train):
        """使用SMOTE处理类别不平衡"""
        # 合并所有特征用于SMOTE
        X_combined = np.concatenate([x.reshape(len(x), -1) for x in X_train], axis=1)
        y_labels = np.argmax(y_train, axis=1)
        
        # 计算各类样本数
        class_counts = np.bincount(y_labels)
        print(f"类别分布 - 阴性类(0): {class_counts[0]}, 阳性类(1): {class_counts[1]}")
        
        # 动态设置采样策略
        # 如果阳性类样本数 < 2000，则设置目标为2000
        # 否则保持原始样本数（即不进行过采样）
        target_samples = min(2000, class_counts[1])
        sampling_strategy = {1: target_samples} if class_counts[1] < 2000 else None
        
        if sampling_strategy is None:
            print("阳性类样本已足够，不进行过采样")
            return X_train, y_train
        
        # 使用SMOTE过采样
        print(f"使用SMOTE过采样，目标阳性类样本数: {target_samples}")
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_res, y_res = smote.fit_resample(X_combined, y_labels)
        
        # 分割回原始格式
        X_train_res = []
        start = 0
        for x in X_train:
            size = np.prod(x.shape[1:])
            X_train_res.append(X_res[:, start:start+size].reshape(-1, *x.shape[1:]))
            start += size
            
        return X_train_res, to_categorical(y_res)
    
    def train_kfold(self):
        """使用K折交叉验证训练模型"""
        print(f"\n=== 开始训练 {self.config.ptm_type}  PTM预测模型 (K折交叉验证) ===")
        
        # 准备数据
        X_all, y_all = self.prepare_training_data() 
        labels = np.argmax(y_all,  axis=1)
        
        # 初始化K折交叉验证
        kfold = StratifiedKFold(n_splits=self.config.k_folds,  shuffle=True, random_state=self.config.random_state) 
        
        # 存储每折的结果 
        fold_results = []
        best_models = []
        
        for fold, (train_idx, test_idx) in enumerate(kfold.split(X_all[0],  labels)):
            self.current_fold  = fold + 1 
            print(f"\n=== 开始第 {self.current_fold}/{self.config.k_folds}  折训练 ===")
            
            # 划分训练集和测试集 
            X_train = [x[train_idx] for x in X_all]
            y_train = y_all[train_idx]
            X_test = [x[test_idx] for x in X_all]
            y_test = y_all[test_idx]
            
            # 进一步划分验证集 
            train_idx, val_idx = train_test_split(
                np.arange(len(X_train[0])), 
                test_size=0.2,
                stratify=np.argmax(y_train,  axis=1),
                random_state=42 
            )
            
            X_val = [x[val_idx] for x in X_train]
            X_train = [x[train_idx] for x in X_train]
            y_val = y_train[val_idx]
            y_train = y_train[train_idx]
            
            # 处理类别不平衡 
            X_train, y_train = self.handle_class_imbalance(X_train,  y_train)
            
            # 创建模型 
            tf.keras.backend.clear_session() 
            self.model  = create_ptm_model(self.config) 
            
            # 编译模型 
            self.model.compile( 
                optimizer=AdamW(learning_rate=self.config.initial_learning_rate, 
                              weight_decay=self.config.weight_decay), 
                loss=self.focal_loss, 
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc')] 
            )
            
            # 回调函数
            callbacks = [
                LearningRateScheduler(self.lr_scheduler), 
                EarlyStopping(
                    monitor='val_auc',
                    patience=self.config.patience, 
                    min_delta=0.01,
                    mode='max',
                    restore_best_weights=True 
                ),
                ModelCheckpoint(
                    os.path.join(self.config.model_dir,  f"{self.config.dataset_name}_{self.config.epochs}_{self.config.patience}_{self.config.batch_size}_fold_{self.current_fold}_best_model.h5"),
                    monitor='val_auc',
                    save_best_only=True,
                    mode='max'
                ),
                CSVLogger(os.path.join(self.config.result_dir,  f"{self.config.dataset_name}_{self.config.epochs}_{self.config.patience}_{self.config.batch_size}_fold_{self.current_fold}_training_log.csv")),
                TensorBoard(log_dir=os.path.join(self.config.result_dir,  f'logs_fold_{self.current_fold}')) 
            ]
            
            # 训练模型
            history = self.model.fit( 
                X_train, y_train,
                batch_size=self.config.batch_size, 
                epochs=self.config.epochs, 
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1 
            )
            
            # 保存训练历史 
            self.plot_training_history(history,  os.path.join(self.config.result_dir,  f"{self.config.dataset_name}_{self.config.epochs}_{self.config.patience}_{self.config.batch_size}_fold_{self.current_fold}_training_history.png")) 
            
            # 加载最佳模型
            self.model  = load_model(
                os.path.join(self.config.model_dir,  f"{self.config.dataset_name}_{self.config.epochs}_{self.config.patience}_{self.config.batch_size}_fold_{self.current_fold}_best_model.h5"), 
                custom_objects={
                    'PositionEmbedding': PositionEmbedding,
                    'MultiHeadSelfAttention': MultiHeadSelfAttention,
                    'TransformerBlock': TransformerBlock,
                    'focal_loss': self.focal_loss 
                }
            )
        
            # 优化阈值
            if self.config.ptm_type == 'phosphorylation':
                best_threshold = self.optimize_threshold_high_specificity(self.model, X_val, y_val)
                print(f"\n[磷酸化] 使用高特异性阈值优化策略")
            if self.config.ptm_type == 'n_linked_glycosylation':
                best_threshold = self.optimize_threshold_f1(self.model, X_val, y_val)
                print(f"\n[磷酸化] 使用高特异性阈值优化策略")
            else:
                best_threshold = self.optimize_threshold_high_specificity2(self.model, X_val, y_val)
                print(f"\n[{self.config.ptm_type}] 使用F1最大化阈值优化策略")
            
            print(f"最佳分类阈值: {best_threshold:.4f}")
            
            # 评估测试集
            print("\n=== 测试集评估 ===")
            test_metrics = self.evaluate_model(self.model,  X_test, y_test, best_threshold)
            
            # 保存结果
            self.save_results(test_metrics,  y_test, X_test, best_threshold, fold=self.current_fold) 
            
            # 存储结果和最佳模型
            fold_results.append(test_metrics) 
            best_models.append(self.model) 
            
            # 释放内存
            tf.keras.backend.clear_session() 
            del self.model 
        
        # 计算平均性能指标 
        avg_metrics = self.calculate_average_metrics(fold_results) 
        
        # 保存平均结果
        self.save_average_results(avg_metrics) 
        
        return best_models, fold_results, avg_metrics
    
    def calculate_average_metrics(self, fold_results):
        """计算K折交叉验证的平均性能指标及标准差 
        
        参数:
            fold_results (list): 包含各折评估结果的字典列表
            
        返回:
            dict: 包含平均指标和标准差（带±符号）的字典 
        """
        if not fold_results:
            raise ValueError("fold_results不能为空列表")
        
        # 基础指标计算 
        avg_metrics = {
            'Accuracy': {
                'mean': np.mean([res['Accuracy'] for res in fold_results]),
                'std': np.std([res['Accuracy'] for res in fold_results])
            },
            'Precision': {
                'mean': np.mean([res['Precision'] for res in fold_results]),
                'std': np.std([res['Precision'] for res in fold_results])
            },
            'Recall': {
                'mean': np.mean([res['Recall'] for res in fold_results]),
                'std': np.std([res['Recall'] for res in fold_results])
            },
            'F1': {
                'mean': np.mean([res['F1'] for res in fold_results]),
                'std': np.std([res['F1'] for res in fold_results])
            },
            'MCC': {
                'mean': np.mean([res['MCC'] for res in fold_results]),
                'std': np.std([res['MCC'] for res in fold_results])
            },
            'AUC': {
                'mean': np.mean([res['AUC'] for res in fold_results]),
                'std': np.std([res['AUC'] for res in fold_results])
            },
            'AUPR': {
                'mean': np.mean([res['AUPR'] for res in fold_results]),
                'std': np.std([res['AUPR'] for res in fold_results])
            },
            'FP_Rate': {
                'mean': np.mean([res['FP Rate'] for res in fold_results]),
                'std': np.std([res['FP Rate'] for res in fold_results])
            },
            'FN_Rate': {
                'mean': np.mean([res['FN Rate'] for res in fold_results]),
                'std': np.std([res['FN Rate'] for res in fold_results])
            }
        }
        
        # 先收集所有键，避免在迭代时修改字典
        metrics_keys = list(avg_metrics.keys())
        
        # 添加格式化字符串版本 
        for metric in metrics_keys:
            avg_metrics[f"{metric}_str"] = \
                f"{avg_metrics[metric]['mean']:.4f} ± {avg_metrics[metric]['std']:.4f}"
        
        # 计算混淆矩阵平均值
        avg_conf_matrix = np.mean( 
            [np.array(res['Confusion Matrix']) for res in fold_results],
            axis=0 
        ).round(2).tolist()
        avg_metrics['Confusion_Matrix_avg'] = avg_conf_matrix 
        
        # 添加各折详细结果引用 
        avg_metrics['fold_details'] = fold_results 
        
        return avg_metrics
    def save_results(self, metrics, y_test, X_test, threshold, fold=1):
        """保存评估结果"""
        result_file = os.path.join(self.config.result_dir,  
                                 f"{self.config.dataset_name}_{self.config.epochs}_{self.config.patience}_{self.config.batch_size}_fold_{fold}_evaluation_results.txt") 
    
        with open(result_file, 'w') as f:
            f.write(f"===  {self.config.ptm_type}  PTM预测结果 (第{fold}折) ===\n\n")
            f.write("===  性能指标 ===\n")
            f.write(f"Accuracy:  {metrics['Accuracy']:.4f}\n")
            f.write(f"Precision:  {metrics['Precision']:.4f}\n")
            f.write(f"Recall:  {metrics['Recall']:.4f}\n")
            f.write(f"F1:  {metrics['F1']:.4f}\n")
            f.write(f"MCC:  {metrics['MCC']:.4f}\n")
            f.write(f"AUC:  {metrics['AUC']:.4f}\n")
            f.write(f"AUPR:  {metrics['AUPR']:.4f}\n")
            f.write(f"FP Rate: {metrics['FP Rate']:.4f}\n")
            f.write(f"FN Rate: {metrics['FN Rate']:.4f}\n\n")
    
            f.write("===  分类报告 ===\n")
            f.write(classification_report( 
                np.argmax(y_test,  axis=1),
                (self.model.predict(X_test)[:,  1] > threshold).astype(int),
                target_names=['Negative', 'Positive']
            ))
    
        # 保存JSON格式的完整结果 
        json_file = os.path.join(self.config.result_dir,  
                               f"{self.config.dataset_name}_{self.config.epochs}_{self.config.patience}_{self.config.batch_size}_fold_{fold}_evaluation_metrics.json") 
    
        # 转换metrics中的NumPy类型为Python原生类型 
        def convert_numpy_types(obj):
            if isinstance(obj, (np.int32,  np.int64)): 
                return int(obj)
            elif isinstance(obj, (np.float32,  np.float64)): 
                return float(obj)
            elif isinstance(obj, np.ndarray): 
                return obj.tolist() 
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()} 
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            else:
                return obj 
    
        metrics_serializable = convert_numpy_types(metrics)
    
        with open(json_file, 'w') as f:
            json.dump(metrics_serializable,  f, indent=4)
    
        print(f"\n=== 第{fold}折测试结果 ===")
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall: {metrics['Recall']:.4f}")
        print(f"F1: {metrics['F1']:.4f}")
        print(f"MCC: {metrics['MCC']:.4f}")
        print(f"AUC: {metrics['AUC']:.4f}")
        print(f"AUPR: {metrics['AUPR']:.4f}")
        print(f"FP Rate: {metrics['FP Rate']:.4f}")
        print(f"FN Rate: {metrics['FN Rate']:.4f}")
        print("\n分类报告:")
        print(classification_report(
            np.argmax(y_test,  axis=1),
            (self.model.predict(X_test)[:,  1] > threshold).astype(int),
            target_names=['Negative', 'Positive']
        ))
    
    def save_average_results(self, avg_metrics):
        """保存K折交叉验证的平均结果"""
        result_file = os.path.join(self.config.result_dir,  
                                 f"{self.config.dataset_name}_{self.config.epochs}_{self.config.patience}_{self.config.batch_size}_average_results.txt") 
    
        with open(result_file, 'w') as f:
            f.write(f"===  {self.config.ptm_type}  PTM预测模型 {self.config.k_folds} 折交叉验证平均结果 ===\n\n")
            f.write("===  平均性能指标 ===\n")
            f.write(f"Accuracy: {avg_metrics['Accuracy']['mean']:.4f}\n")
            f.write(f"Precision: {avg_metrics['Precision']['mean']:.4f}\n")
            f.write(f"Recall: {avg_metrics['Recall']['mean']:.4f}\n")
            f.write(f"F1: {avg_metrics['F1']['mean']:.4f}\n")
            f.write(f"MCC: {avg_metrics['MCC']['mean']:.4f}\n")
            f.write(f"AUC: {avg_metrics['AUC']['mean']:.4f}\n")
            f.write(f"AUPR: {avg_metrics['AUPR']['mean']:.4f}\n")
            f.write(f"FP Rate: {avg_metrics['FP_Rate']['mean']:.4f}\n")
            f.write(f"FN Rate: {avg_metrics['FN_Rate']['mean']:.4f}\n")
