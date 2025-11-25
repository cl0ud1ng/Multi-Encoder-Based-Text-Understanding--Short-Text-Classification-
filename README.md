# 文本分类项目 - 基于深度学习的新闻分类

本项目实现了基于深度学习的中文文本分类任务，使用 PyTorch 框架，支持 BiLSTM 和 Transformer 两种神经网络模型对新闻文本进行分类。

## 📋 项目概述

本项目针对 TNews 公开数据集进行新闻文本分类，通过深度学习模型自动识别新闻所属类别。项目提供了完整的数据处理、模型训练、验证和预测流程。

## 🗂️ 项目结构

```
hw3/
├── Exp_DataSet.py    # 数据集处理模块
├── Exp_Model.py      # 模型定义模块
├── Exp_Train.py      # 训练和预测主程序
├── data/             # 数据目录（需自行创建）
│   └── tnews_public/
│       ├── labels.json    # 标签映射文件
│       ├── train.json     # 训练集
│       ├── dev.json       # 验证集
│       └── test.json      # 测试集
└── output/           # 输出目录（自动创建）
    ├── model.ckpt         # 保存的最优模型
    └── predict.json       # 测试集预测结果
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
双向长短期记忆网络模型，适合序列数据的分类任务。

**模型架构**：
- Embedding 层：将token转换为词向量
- BiLSTM 层：双向LSTM提取序列特征
- 分类器：需自行设计（待完成部分）

**参数说明**：
- `vocab_size`: 词汇表大小
- `ntoken`: 每个句子的token数量
- `d_emb`: 词向量维度（默认100）
- `d_hid`: LSTM隐藏层维度（默认80）
- `nlayers`: LSTM层数（默认1）
- `dropout`: Dropout比率（默认0.2）
- `embedding_weight`: 预训练词向量（可选）

#### Transformer_model 类
基于 Transformer 编码器的文本分类模型。

**模型架构**：
- Embedding 层：词嵌入
- PositionalEncoding：位置编码
- TransformerEncoder：多层Transformer编码器
- 分类器：需自行设计（待完成部分）

**参数说明**：
- `vocab_size`: 词汇表大小
- `ntoken`: 序列长度
- `d_emb`: 嵌入维度（默认512）
- `d_hid`: 前馈网络维度（默认2048）
- `nhead`: 多头注意力头数（默认8）
- `nlayers`: Transformer层数（默认6）
- `dropout`: Dropout比率（默认0.2）
- `embedding_weight`: 预训练词向量（可选）

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

## ⚙️ 超参数配置

可在 `Exp_Train.py` 中修改以下超参数：

```python
embedding_dim = 100        # 词向量维度
max_token_per_sent = 50    # 每个句子的最大token数
batch_size = 16            # 批次大小
num_epochs = 5             # 训练轮数
lr = 1e-4                  # 学习率
```

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

在 `Exp_Train.py` 中可以切换模型：

**使用 BiLSTM 模型**：
```python
model = BiLSTM_model(vocab_size=vocab_size, ntoken=max_token_per_sent, d_emb=embedding_dim).to(device)
```

**使用 Transformer 模型**：
```python
model = Transformer_model(vocab_size=vocab_size, ntoken=max_token_per_sent, d_emb=embedding_dim).to(device)
```

## 📊 输出结果

训练过程中会实时显示：
- 训练损失 (train loss)
- 训练准确率 (train accuracy)
- 验证准确率 (valid accuracy)

最终生成：
- `output/model.ckpt`: 验证集上表现最好的模型
- `output/predict.json`: 测试集预测结果

## 🔨 待完成部分

代码中标记了 `#-----begin-----#` 和 `#------end------#` 的部分需要根据具体任务补充：

### 任务一 & 任务二（基础模型）
1. **BiLSTM_model**：
   - 设计隐藏层输出的处理方法
   - 实现分类器（全连接层）

2. **Transformer_model**：
   - 设计Transformer编码器输出的处理策略
   - 实现分类器

### 任务三（预训练模型）
1. **Exp_DataSet.py**：
   - 在 `Corpus.__init__()` 中加载预训练词向量
   - 创建 embedding_weight 矩阵
   - 在 `tokenize()` 中使用预训练分词器

2. **模型初始化**：
   - 传入 embedding_weight 参数

## 💡 开发建议

### 分类器设计
建议在模型的 forward 方法中添加：
- 池化层（如取最后一个时间步、平均池化、最大池化）
- 全连接层进行分类
- 输出层（logits，不需要 softmax，CrossEntropyLoss 会处理）

### 预训练词向量
如果使用预训练词向量：
1. 加载词向量文件（如 Word2Vec, GloVe）
2. 为词汇表中每个词查找对应向量
3. 处理 [PAD] 和 [UNK] 特殊token
4. 构建 embedding_weight 矩阵

### 性能优化建议
- 调整学习率和批次大小
- 尝试不同的模型架构参数
- 使用学习率调度器
- 添加早停机制
- 数据增强

## 🎯 技术特点

- ✅ 模块化设计，代码结构清晰
- ✅ 支持 GPU 加速训练
- ✅ 灵活的模型选择机制
- ✅ 完整的训练-验证-测试流程
- ✅ 自动保存最优模型
- ✅ 支持预训练词向量
- ✅ 实时训练进度显示

## 📝 注意事项

1. 确保数据文件格式正确，编码为 UTF-8
2. 首次运行会自动构建词汇表
3. 模型保存在 `output/` 目录，如不存在会自动创建
4. 测试集的标签字段存储的是数据ID，用于生成预测文件
5. 分类器部分需要根据具体任务自行实现

## 📄 License

本项目为教学实验项目，仅供学习使用。

---

**作者**：NLP课程作业  
**更新时间**：2025年11月
