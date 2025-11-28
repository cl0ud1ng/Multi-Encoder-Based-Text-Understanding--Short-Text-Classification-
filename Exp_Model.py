import torch.nn as nn
import torch as torch
import math
from transformers import BertModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer_model(nn.Module):
    def __init__(self, vocab_size, ntoken, d_emb=512, d_hid=2048, nhead=8, nlayers=6, dropout=0.2, num_classes=15, embedding_weight=None):
        super(Transformer_model, self).__init__()
        # 将"预训练的词向量"整理成 token->embedding 的二维映射矩阵 emdedding_weight 的形式，初始化 _weight
        # 当 emdedding_weight == None 时，表示随机初始化
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_emb, _weight=embedding_weight)

        self.pos_encoder = PositionalEncoding(d_model=d_emb, max_len=ntoken)
        self.encode_layer = nn.TransformerEncoderLayer(d_model=d_emb, nhead=nhead, dim_feedforward=d_hid)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encode_layer, num_layers=nlayers)
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 请自行设计对 transformer 隐藏层数据的处理和选择方法
        self.dropout = nn.Dropout(dropout)  # 可选

        # 请自行设计分类器
        # 使用平均池化后通过全连接层进行分类
        self.classifier = nn.Linear(d_emb, num_classes)

        #------------------------------------------------------end------------------------------------------------------#

    def forward(self, x):
        # x: [batch_size, seq_len], token ids
        padding_mask = (x == 0)  # [batch_size, seq_len], True for padding positions
        
        x = self.embed(x)     
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, d_emb]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)  # 传入 padding mask
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, d_emb]
        
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 对 transformer_encoder 的隐藏层输出进行处理和选择，并完成分类
        # 使用平均池化聚合所有时间步的信息，忽略 padding 位置
        mask = (~padding_mask).float().unsqueeze(-1)  # [batch_size, seq_len, 1]
        x = x * mask
        x = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # 平均池化，避免除零
        
        x = self.dropout(x)
        x = self.classifier(x)  # 分类器输出，形状: [batch_size, num_classes]

        #------------------------------------------------------end------------------------------------------------------#
        return x
    
    
class BiLSTM_model(nn.Module):
    def __init__(self, vocab_size, ntoken, d_emb=100, d_hid=80, nlayers=1, dropout=0.2, num_classes=15, embedding_weight=None):
        super(BiLSTM_model, self).__init__()
        # 将"预训练的词向量"整理成 token->embedding 的二维映射矩阵 emdedding_weight 的形式，初始化 _weight
        # 当 emdedding_weight == None 时，表示随机初始化
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_emb, _weight=embedding_weight)

        self.lstm = nn.LSTM(input_size=d_emb, hidden_size=d_hid, num_layers=nlayers, bidirectional=True, batch_first=True)
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 请自行设计对 bilstm 隐藏层数据的处理和选择方法
        self.dropout = nn.Dropout(dropout)  # 可选

        # 请自行设计分类器
        # 双向LSTM输出维度是 d_hid * 2，这里使用最后一个时间步的输出
        self.classifier = nn.Linear(d_hid * 2, num_classes)

        #------------------------------------------------------end------------------------------------------------------#

    def forward(self, x):
        # x: [batch_size, seq_len], token ids
        mask = (x != 0).float()  # [batch_size, seq_len], 1 for real tokens, 0 for padding
        
        x = self.embed(x)  # [batch_size, seq_len, d_emb]
        x, _ = self.lstm(x)  # [batch_size, seq_len, d_hid*2]
        
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 对 bilstm 的隐藏层输出进行处理和选择，并完成分类
        # 使用平均池化，忽略 padding 位置
        mask = mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        x = x * mask  # 将 padding 位置的输出置零
        x = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # 平均池化，避免除零
        
        x = self.dropout(x)
        x = self.classifier(x)  # 分类器输出，形状: [batch_size, num_classes]

        #------------------------------------------------------end------------------------------------------------------#
        return x


class BERT_model(nn.Module):
    """
    基于预训练 BERT 的文本分类模型。
    冻结 BERT 预训练层，只训练分类层。
    """
    def __init__(self, bert_model_name='bert-base-chinese', num_classes=15, dropout=0.1, freeze_bert=True):
        super(BERT_model, self).__init__()
        
        # 加载预训练 BERT 模型
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # 冻结 BERT 预训练层的参数
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # 获取 BERT 隐藏层维度
        hidden_size = self.bert.config.hidden_size  # 768 for bert-base
        
        # 分类层（只训练这一部分）
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, num_classes)
                )
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Args:
            input_ids: [batch_size, seq_len] - token IDs
            attention_mask: [batch_size, seq_len] - 1 for real tokens, 0 for padding
            token_type_ids: [batch_size, seq_len] - segment IDs (optional)
        Returns:
            logits: [batch_size, num_classes]
        """
        # BERT 输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 使用 [CLS] token 的输出作为句子表示
        pooled_output = outputs.pooler_output
        
        # 分类
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        
        return logits