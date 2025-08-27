import os
import numpy as np
import json
from config import Config
from data_loader import PTMDataLoader
from trainer import PTMTrainer
# ==================== 主程序 ====================
if __name__ == "__main__":
    # 选择要训练的PTM类型
    PTM_TYPES = [
       'phosphorylation',
        'acetylation',
        'ubiquitination',
        'hydroxylation',
        'gamma_carboxyglutamic_acid',
        'lys_methylation',
        'malonylation',
        'arg_methylation',
        'crotonylation',
        'succinylation',
        'glutathionylation',
        'sumoylation',
        's_nitrosylation',
        'glutarylation',
        'citrullination',
        'o_linked_glycosylation',
        'n_linked_glycosylation'
    ]
    all_results = {}  # 用于存储所有PTM类型的结果
    
    for ptm_type in PTM_TYPES:
        print(f"\n{'='*50}")
        print(f"正在处理: {ptm_type.upper()}  PTM预测 (K折交叉验证)")
        print(f"{'='*50}")
 
        # 创建配置和训练器 
        config = Config(ptm_type=ptm_type)
        trainer = PTMTrainer(config)
 
        # 训练和评估模型 
        best_models, fold_results, avg_metrics = trainer.train_kfold() 
 
        # 保存平均结果 
        all_results[ptm_type] = {
            'fold_results': fold_results,
            'average_metrics': avg_metrics 
        }
 
    # 统一展示所有PTM类型的评估结果 
    print("\n\n=== 所有PTM类型评估结果汇总 ===")
    for ptm_type, results in all_results.items(): 
        avg_metrics = results['average_metrics']
        print(f"\n{ptm_type.upper()}  平均结果:")
        print(f"Accuracy: {avg_metrics['Accuracy']:.4f}")
        print(f"Precision: {avg_metrics['Precision']:.4f}")
        print(f"Recall: {avg_metrics['Recall']:.4f}")
        print(f"F1: {avg_metrics['F1']:.4f}")
        print(f"MCC: {avg_metrics['MCC']:.4f}")
        print(f"AUC: {avg_metrics['AUC']:.4f}")
        print(f"AUPR: {avg_metrics['AUPR']:.4f}")
        print(f"FP Rate: {avg_metrics['FP Rate']:.4f}")
        print(f"FN Rate: {avg_metrics['FN Rate']:.4f}")
 
    # 保存汇总结果到文件 
    summary_file = os.path.join(config.root_dir,  
                              f"{config.dataset_name}_{config.epochs}_{config.patience}_{config.batch_size}_PTM_kfold_results_summary.txt") 
    with open(summary_file, 'w') as f:
        f.write("===  PTM预测模型评估结果汇总 (K折交叉验证) ===\n\n")
        for ptm_type, results in all_results.items(): 
            avg_metrics = results['average_metrics']
            f.write(f"{ptm_type.upper()}  平均结果:\n")
            f.write(f"Accuracy:  {avg_metrics['Accuracy']:.4f}\n")
            f.write(f"Precision:  {avg_metrics['Precision']:.4f}\n")
            f.write(f"Recall:  {avg_metrics['Recall']:.4f}\n")
            f.write(f"F1:  {avg_metrics['F1']:.4f}\n")
            f.write(f"MCC:  {avg_metrics['MCC']:.4f}\n")
            f.write(f"AUC:  {avg_metrics['AUC']:.4f}\n")
            f.write(f"AUPR:  {avg_metrics['AUPR']:.4f}\n")
            f.write(f"FP Rate: {avg_metrics['FP Rate']:.4f}\n")
            f.write(f"FN Rate: {avg_metrics['FN Rate']:.4f}\n\n")