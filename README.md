### PTM（翻译后修饰）位点预测工具:

### 

### 基于深度学习的蛋白质翻译后修饰位点预测框架，支持磷酸化、乙酰化、泛素化等多种修饰类型预测

### 

### 核心功能：

###   1.多类型PTM预测（支持16种修饰类型）

###   2.混合模型架构（Transformer + CNN）

###   3.集成ESM-2蛋白质语言模型

###   4.结合结构特征（二级结构、溶剂可及性等）

###   5.支持SMOTE不平衡数据处理

### 

### 环境安装配置：

###   conda env create -f environment.yml

###   conda activate ptm-env

###   如果使用GPU，就直接选择environment.yml，使用cpu的话可以删除GPU相关依赖

###   

### 硬件要求：

###   1.GPU：支持CUDA 11.8+的NVIDIA显卡（推荐）

###   2.内存：16GB以上（处理ESM-2特征需要）

###   3.磁盘：至少50GB空间（存储模型和特征）

### 代码使用前准备：
###   一些大文件（ESM2生成的特征文件，数据集）需要单独下载，下载文件以及链接如下，下载后的存放位置也一并给出。
### ESM2特征文件（ptm_data_esm_features.npy下载地址：https://drive.google.com/file/d/1wJgUQ861iqM3CXJJoQb6AT_jWT-2Dedi/view?usp=drive_link），放在目录deepPTMpred\pred下。
### 数据集（ptm_data.csv下载地址：https://drive.google.com/file/d/1sBSODTVUOm3Q7wa05fFmupYiQs072nKX/view?usp=drive_link），放在deepPTMpred\data目录下。
### esm模型文件（下载地址：https://drive.google.com/drive/folders/1KYbfh3PGRhd_s0wn-8tZcbX_uvo1xdNm?usp=drive_link），文件夹直接放在deepPTMpred下即可。

### 代码使用说明：

### &nbsp; 训练模型时，可以选择训练PTM的类型包括以下类型：磷酸化(phosphorylation)、乙酰化(acetylation)、泛素化(ubiquitination)、羟基化(hydroxylation)、γ-羧基谷氨酸(gamma\_carboxyglutamic\_acid)、赖氨酸甲基化(lys\_methylation)、丙二酰化(malonylation)、精氨酸甲基化(arg\_methylation)、巴豆酰化(crotonylation)、琥珀酰化(succinylation)、谷胱甘肽化(glutathionylation)、SUMO化(sumoylation)、S-亚硝基化(s\_nitrosylation)、戊二酰化(glutarylation)、瓜氨酸化(citrullination)、O-糖基化(o\_linked\_glycosylation)、N-糖基化(n\_linked\_glycosylation)



### &nbsp; 如果你想使用自己的数据集去训练，那么你需要将自己的数据集放入data文件下，使用e2\_data.py生成ESM特征文件,随后使用python main.py训练（要保证数据集的标签一致）。

### &nbsp; 如果你想复现项目的话，在下载项目之后，你需要先配置环境，随后修改config.py中的路径配置，之后运行python main.py即可。

###   如果你想使用训练好的模型预测，那么你需要先运行esm2\_feature.py生成esm特征文件，再运行python predict.py，只需要修改ptm\_type（想要预测的ptm类型），model\_path（训练好的模型权重文件）protein\_id（蛋白质ID），protein\_sequence（蛋白质序列）， pdb\_path这5个参数即可。项目中给出了一个关于与阿尔兹海默症有关的蛋白质P10636磷酸化预测的案例，如果要测试的话，在配置环境完成后，直接运行predict.py即可。



### 文件结构：

### deepPTMpred/

### ├── pred/

### │   ├──ptm\_data\_esm\_features.npy   #ptm\_data数据集生成的ESM2特征权重文件

### │   ├──ptm\_data\_metadata.npy

### │   ├── custom\_esm/

### │       ├── P10636\_full\_esm.npz # 存放ESM特征文件

### │   └── train\_PTM/

### │       ├── config.py       # 参数配置文件

### │       ├── data\_loader.py     # 数据加载与处理

### │       ├── model.py         # 模型架构定义

### │       ├── trainer.py         # 训练流程控制

### │       ├── e2\_single\_data.py  # 单个蛋白质ESM2特征文件生成

### │       ├── e2\_data.py        #  数据集类ESM2特征文件生成

### │       ├── predict.py        # 预测脚本

### │       ├── main.py            # 主程序入口

### │       ├── environment.yml        # Conda环境配置（GPU/cpu版）

### │       ├── models\_phosphorylation\_esm2/      # 模型保存

### │               ├──ptm\_data\_201\_39\_64\_best\_model.h5

### │       ├── models\_phosphorylation\_esm2\_kfold # K折模型

### │       └── results\_\*/             # 各类结果

### ├── data/

### │   ├── ptm\_data.csv         # 数据集
### │   ├── AF-P10636-F1-model_v4,pdb         # P10636的pdb文件
### │   ├── AF-P31749-F1-model_v4,pdb         # P31749的pdb文件

### ├── esm/

### │   ├── hub/

### │   │   ├── checkpoints/

### │   │   │   ├── esm2\_t33\_650M\_UR50D.pt

### └── README.md                      # 项目总说明






