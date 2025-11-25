# 文本分类项目 - 基于深度学习的新闻分类

本项目实现了基于深度学习的中文文本分类任务，使用 PyTorch 框架，支持 BiLSTM 和 Transformer 两种神经网络模型对新闻文本进行分类。

> **项目状态**：✅ 已完成，可直接运行  
> **预训练词向量**：300维 SGNS (Skip-Gram with Negative Sampling)  
> **当前模型**：BiLSTM

## 📋 项目概述

本项目针对 TNews 公开数据集进行新闻文本分类，通过深度学习模型自动识别新闻所属类别。项目提供了完整的数据处理、模型训练、验证和预测流程，并集成了300维预训练词向量。

## 🚀 快速开始

### 一键运行
```bash
# 1. 确保数据集已准备好（放在 ./data/tnews_public/ 目录下）
# 2. 直接运行训练脚本
python Exp_Train.py
```

程序将自动完成训练、验证和测试，最终在 `./output/` 目录生成模型和预测结果。

### 基本流程
1. **数据加载** → 自动从 JSON 文件读取并构建词汇表
2. **模型训练** → 5个epoch，每个epoch显示训练进度
3. **模型验证** → 每个epoch后在验证集上评估
4. **保存最优模型** → 自动保存验证集准确率最高的模型
5. **测试集预测** → 使用最优模型生成预测文件

## 🗂️ 项目结构

```
hw3/
├── Exp_DataSet.py                                      # 数据集处理模块
├── Exp_Model.py                                        # 模型定义模块（BiLSTM & Transformer）
├── Exp_Train.py                                        # 训练和预测主程序
├── README.md                                           # 项目说明文档
├── sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5  # 300维预训练词向量
├── data/                                               # 数据目录
│   └── tnews_public/
│       ├── labels.json                                 # 标签映射文件
│       ├── train.json                                  # 训练集
│       ├── dev.json                                    # 验证集
│       └── test.json                                   # 测试集
└── output/                                             # 输出目录（运行时自动创建）
    ├── model.ckpt                                      # 验证集上最优模型
    └── predict.json                                    # 测试集预测结果
```

## 🔧 核心模块说明

### 1. Exp_DataSet.py - 数据处理模块

#### Dictionary 类
- **功能**：管理词汇表和标签映射
- **属性**：
  - `word2tkn`: 词到token的映射字典
  - `tkn2word`: token到词的映射列表
  - `label2idx`: 标签到索引的映射
  - `idx2label`: 索引到标签的映射

#### Corpus 类
- **功能**：完成数据集的读取和预处理
- **主要方法**：
  - `tokenize()`: 将文本转换为token序列
  - `pad()`: 填充序列至固定长度
- **输出**：返回 PyTorch TensorDataset 格式的数据

**特性**：
- 支持自定义最大句子长度
- 自动构建词汇表
- 支持训练集、验证集和测试集的处理
- 预留预训练词向量接口

### 2. Exp_Model.py - 模型定义模块

#### BiLSTM_model 类
双向长短期记忆网络模型，适合序列数据的分类任务。✅ **已完成实现**

**模型架构**：
- **Embedding 层**：将token转换为词向量（支持预训练词向量）
- **BiLSTM 层**：双向LSTM提取序列特征，输出维度为 `d_hid * 2`
- **Dropout 层**：防止过拟合
- **分类器**：全连接层 `Linear(d_hid * 2, num_classes)`
- **输出**：分类logits，形状为 `[batch_size, num_classes]`

**分类策略**：使用最后一个时间步的隐藏状态进行分类

**参数说明**：
- `vocab_size`: 词汇表大小
- `ntoken`: 每个句子的token数量
- `d_emb`: 词向量维度（默认100，当前项目使用300）
- `d_hid`: LSTM隐藏层维度（默认80）
- `nlayers`: LSTM层数（默认1）
- `dropout`: Dropout比率（默认0.2）
- `num_classes`: 类别数量（默认15）
- `embedding_weight`: 预训练词向量矩阵（可选）

#### Transformer_model 类
基于 Transformer 编码器的文本分类模型。✅ **已完成实现**

**模型架构**：
- **Embedding 层**：词嵌入（支持预训练词向量）
- **PositionalEncoding**：位置编码，使模型感知序列位置信息
- **TransformerEncoder**：多层Transformer编码器，多头自注意力机制
- **平均池化**：聚合所有时间步的表示
- **Dropout 层**：防止过拟合
- **分类器**：全连接层 `Linear(d_emb, num_classes)`
- **输出**：分类logits，形状为 `[batch_size, num_classes]`

**分类策略**：对所有时间步的输出进行平均池化后分类

**参数说明**：
- `vocab_size`: 词汇表大小
- `ntoken`: 序列长度
- `d_emb`: 嵌入维度（默认512）
- `d_hid`: 前馈网络维度（默认2048）
- `nhead`: 多头注意力头数（默认8，**注意**: `d_emb` 必须能被 `nhead` 整除）
- `nlayers`: Transformer层数（默认6）
- `dropout`: Dropout比率（默认0.2）
- `num_classes`: 类别数量（默认15）
- `embedding_weight`: 预训练词向量矩阵（可选）

#### PositionalEncoding 类
为 Transformer 模型提供位置编码，使模型能够感知序列中的位置信息。

### 3. Exp_Train.py - 训练主程序

主要包含三个核心函数：

#### train()
- 执行模型训练过程
- 使用 tqdm 显示训练进度
- 自动保存验证集上表现最好的模型
- 输出每个epoch的训练损失和准确率

#### valid()
- 在验证集上评估模型性能
- 返回验证集准确率
- 用于模型选择

#### predict()
- 加载训练好的最优模型
- 对测试集进行预测
- 生成JSON格式的预测结果文件

## ⚙️ 当前超参数配置

在 `Exp_Train.py` 中的超参数设置（第116-122行）：

```python
embedding_dim = 300        # 词向量维度（匹配300维预训练词向量）
max_token_per_sent = 50    # 每个句子的最大token数
batch_size = 16            # 批次大小
num_epochs = 5             # 训练轮数
lr = 1e-4                  # 学习率（Adam优化器）
```

**说明**：
- `embedding_dim=300`：与预训练词向量 `sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5` 的维度匹配
- 优化器：Adam with weight_decay=5e-4
- 损失函数：CrossEntropyLoss

## 🚀 使用方法

### 环境要求

```bash
pip install torch numpy tqdm
```

**依赖包**：
- Python 3.6+
- PyTorch 1.0+
- numpy
- tqdm

### 数据准备

1. 在项目根目录创建 `data/tnews_public/` 文件夹
2. 准备以下数据文件（JSON格式）：
   - `labels.json`: 标签定义文件
   - `train.json`: 训练数据
   - `dev.json`: 验证数据
   - `test.json`: 测试数据

**数据格式示例**：

labels.json:
```json
{"label": "news_story", "label_desc": "故事"}
{"label": "news_culture", "label_desc": "文化"}
```

train.json / dev.json:
```json
{"id": 1, "sentence": "今天天气真好", "label": "news_story"}
```

test.json:
```json
{"id": 1, "sentence": "测试句子"}
```

### 运行训练

```bash
python Exp_Train.py
```

程序将自动完成：
1. 加载和预处理数据
2. 构建词汇表
3. 训练模型
4. 在验证集上评估
5. 保存最优模型
6. 对测试集进行预测

### 模型选择

在 `Exp_Train.py` 第134-136行可以切换模型：

**方案1：使用 BiLSTM 模型（当前配置）**：
```python
model = BiLSTM_model(
    vocab_size=vocab_size, 
    ntoken=max_token_per_sent, 
    d_emb=embedding_dim,  # 300
    num_classes=num_classes
).to(device)
```

**方案2：使用 Transformer 模型**：
```python
model = Transformer_model(
    vocab_size=vocab_size, 
    ntoken=max_token_per_sent, 
    d_emb=300,          # 与embedding_dim匹配
    nhead=6,            # 300÷6=50，可整除
    d_hid=1024,         # 前馈网络维度
    nlayers=4,          # Transformer层数
    num_classes=num_classes
).to(device)
```

**重要提示**：
- 使用Transformer时，必须确保 `d_emb % nhead == 0`
- 对于300维词向量，推荐的 `nhead` 选项：5, 6, 10, 12, 15
- 如使用预训练词向量，需在数据集处理时传入 `embedding_weight` 参数

## 📊 输出结果

训练过程中会实时显示：
- 训练损失 (train loss)
- 训练准确率 (train accuracy)
- 验证准确率 (valid accuracy)

最终生成：
- `output/model.ckpt`: 验证集上表现最好的模型
- `output/predict.json`: 测试集预测结果

## 🔧 预训练词向量说明

### 词向量文件
- **文件名**：`sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5`
- **算法**：SGNS (Skip-Gram with Negative Sampling)
- **维度**：300
- **训练参数**：
  - 动态窗口大小：5
  - 阈值：10
  - 负采样数：5
  - 迭代次数：5

### 使用预训练词向量（可选）

如果要在模型中使用预训练词向量，需要在 `Exp_DataSet.py` 中补充加载逻辑：

1. **在 `Corpus.__init__()` 中加载词向量**（第46-51行）：
```python
# 加载预训练词向量文件
embedding_weight = self.load_pretrained_embeddings(
    embedding_path='sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5',
    embedding_dim=300
)
```

2. **实现加载函数**：
```python
def load_pretrained_embeddings(self, embedding_path, embedding_dim):
    # 创建词向量矩阵
    embedding_weight = np.random.randn(len(self.dictionary.tkn2word), embedding_dim)
    
    # 为[PAD]设置零向量
    embedding_weight[0] = np.zeros(embedding_dim)
    
    # 加载预训练向量并填充矩阵
    # ... (具体实现)
    
    return torch.FloatTensor(embedding_weight)
```

3. **在模型初始化时传入**：
```python
model = BiLSTM_model(
    vocab_size=vocab_size,
    ntoken=max_token_per_sent,
    d_emb=embedding_dim,
    num_classes=num_classes,
    embedding_weight=dataset.embedding_weight  # 传入预训练权重
).to(device)
```

### 性能优化建议

**超参数调整**：
- 调整学习率（当前 1e-4）
- 修改批次大小（当前 16）
- 增加训练轮数（当前 5）

**模型优化**：
- 使用学习率调度器（如 ReduceLROnPlateau）
- 添加早停机制（Early Stopping）
- 尝试不同的池化策略（最大池化、注意力池化）
- 调整 Dropout 比率

**数据增强**：
- 同义词替换
- 随机删除
- 回译（Back Translation）

## 🎯 技术特点

### 已实现功能
- ✅ **完整的模型实现**：BiLSTM 和 Transformer 均已实现分类器
- ✅ **模块化设计**：数据处理、模型定义、训练分离
- ✅ **GPU 加速**：自动检测并使用 CUDA
- ✅ **双模型支持**：可轻松切换 BiLSTM 和 Transformer
- ✅ **自动模型选择**：保存验证集上表现最好的模型
- ✅ **预训练词向量接口**：支持加载外部词向量（300维 SGNS）
- ✅ **实时进度显示**：tqdm 进度条显示训练状态
- ✅ **完整流程**：训练-验证-测试一体化

### 模型特性
- **BiLSTM**：使用最后时间步分类策略
- **Transformer**：使用平均池化分类策略
- **自动获取类别数**：无需手动指定分类数量
- **灵活的超参数配置**：集中管理，易于调整

## 📝 注意事项

### 运行前检查
1. ✅ 确保数据文件格式正确，编码为 UTF-8
2. ✅ 数据集路径：`./data/tnews_public/`
3. ✅ 输出目录会自动创建：`./output/`
4. ✅ 首次运行会自动构建词汇表

### 重要提示
- **embedding_dim 必须匹配**：当前配置为 300，与预训练词向量维度一致
- **Transformer 维度约束**：使用 Transformer 时，`d_emb` 必须能被 `nhead` 整除
- **测试集标签**：测试集的 label 字段存储数据 ID（用于生成预测文件）
- **模型保存**：仅保存验证集上表现最好的模型

### 常见问题
**Q: 如何切换模型？**  
A: 修改 `Exp_Train.py` 第135-136行的模型初始化代码

**Q: 如何使用预训练词向量？**  
A: 在 `Exp_DataSet.py` 中加载词向量文件，生成 `embedding_weight` 矩阵并传递给模型

**Q: Transformer 报错 "embed_dim must be divisible by num_heads"？**  
A: 确保 `d_emb % nhead == 0`，对于 300 维，推荐 `nhead=6`

**Q: GPU 内存不足？**  
A: 减小 `batch_size` 或减少模型层数（`nlayers`）

## 📄 License

本项目为教学实验项目，仅供学习使用。

---

**作者**：NLP课程作业  
**更新时间**：2025年11月
