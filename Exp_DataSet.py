import os
import json
import pickle
import numpy as np
import torch
import jieba
from torch.utils.data import TensorDataset

class Dictionary(object):
    def __init__(self, path):

        self.word2tkn = {"[PAD]": 0, "[UNK]": 1}
        self.tkn2word = ["[PAD]", "[UNK]"]

        self.label2idx = {}
        self.idx2label = []

        # 获取 label 的 映射
        with open(os.path.join(path, 'labels.json'), 'r', encoding='utf-8') as f:
            for line in f:
                one_data = json.loads(line)
                label, label_desc = one_data['label'], one_data['label_desc']
                self.idx2label.append([label, label_desc])
                self.label2idx[label] = len(self.idx2label) - 1

    def add_word(self, word):
        if word not in self.word2tkn:
            self.tkn2word.append(word)
            self.word2tkn[word] = len(self.tkn2word) - 1
        return self.word2tkn[word]


class Corpus(object):
    '''
    完成对数据集的读取和预处理，处理后得到所有文本数据的对应的 token 表示及相应的标签。
    
    该类适用于任务一、任务二，若要完成任务三，需对整个类进行调整，例如，可直接调用预训练模型提供的 tokenizer 将文本转为对应的 token 序列。
    '''
    def __init__(self, path, max_token_per_sent, stopwords_path='./stopwords.txt'):
        self.dictionary = Dictionary(path)

        self.max_token_per_sent = max_token_per_sent
        
        # 加载停用词
        self.stopwords = set()
        if stopwords_path and os.path.exists(stopwords_path):
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.stopwords.add(line.strip())
            print(f"Loaded {len(self.stopwords)} stopwords")

        self.train = self.tokenize(os.path.join(path, 'train.json'))
        self.valid = self.tokenize(os.path.join(path, 'dev.json'))
        self.test = self.tokenize(os.path.join(path, 'test.json'), True)

        #-----------------------------------------------------begin-----------------------------------------------------#
        # 若要采用预训练的 embedding, 需处理得到 token->embedding 的映射矩阵 embedding_weight。矩阵的格式参考 nn.Embedding() 中的参数 _weight
        # 注意，需考虑 [PAD] 和 [UNK] 两个特殊词向量的设置
        self.embedding_weight = None  # 默认不使用预训练词向量

        #------------------------------------------------------end------------------------------------------------------#

    def load_pretrained_embedding(self, embedding_path, embedding_dim=300):
        '''
        加载预训练词向量，构建 embedding_weight 矩阵
        '''
        print("Loading pretrained embeddings...")
        
        # 先读取预训练词向量到字典
        pretrained = {}
        with open(embedding_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()  # 跳过第一行（词汇数和维度）
            for line in f:
                parts = line.strip().split()
                if len(parts) > embedding_dim:
                    word = parts[0]
                    vector = np.array([float(x) for x in parts[1:embedding_dim+1]])
                    pretrained[word] = vector
        
        # 构建 embedding_weight 矩阵
        vocab_size = len(self.dictionary.tkn2word)
        self.embedding_weight = np.zeros((vocab_size, embedding_dim))
        
        # [PAD] 用零向量，[UNK] 用随机向量
        self.embedding_weight[1] = np.random.uniform(-0.25, 0.25, embedding_dim)
        
        found = 0
        oov_but_recovered = 0
        for idx, word in enumerate(self.dictionary.tkn2word):
            if word in pretrained:
                self.embedding_weight[idx] = pretrained[word]
                found += 1
            elif len(word) > 1:
                # 对于未覆盖的多字词，尝试用字符向量的平均值
                char_vectors = [pretrained[c] for c in word if c in pretrained]
                if char_vectors:
                    self.embedding_weight[idx] = np.mean(char_vectors, axis=0)
                    oov_but_recovered += 1
                else:
                    self.embedding_weight[idx] = np.random.uniform(-0.25, 0.25, embedding_dim)
            else:
                # 单字且不在词表中，随机初始化
                self.embedding_weight[idx] = np.random.uniform(-0.25, 0.25, embedding_dim)
        
        print(f"Found {found}/{vocab_size} words in pretrained embeddings ({100*found/vocab_size:.1f}%)")
        print(f"Recovered {oov_but_recovered} OOV words using character vectors")
        self.embedding_weight = torch.tensor(self.embedding_weight, dtype=torch.float32)
        return self.embedding_weight

    def save_preprocessed(self, save_path):
        '''
        保存预处理后的数据（包括词典、数据集、词向量）
        '''
        data = {
            'dictionary': self.dictionary,
            'train': self.train,
            'valid': self.valid,
            'test': self.test,
            'embedding_weight': self.embedding_weight,
            'max_token_per_sent': self.max_token_per_sent
        }
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Preprocessed data saved to {save_path}")

    @staticmethod
    def load_preprocessed(load_path):
        '''
        加载预处理后的数据，返回 Corpus 对象
        '''
        print(f"Loading preprocessed data from {load_path}...")
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        # 创建一个空的 Corpus 对象并填充数据
        corpus = object.__new__(Corpus)
        corpus.dictionary = data['dictionary']
        corpus.train = data['train']
        corpus.valid = data['valid']
        corpus.test = data['test']
        corpus.embedding_weight = data['embedding_weight']
        corpus.max_token_per_sent = data['max_token_per_sent']
        print("Preprocessed data loaded successfully!")
        return corpus

    def pad(self, origin_token_seq):
        '''
        padding: 将原始的 token 序列补 0 至预设的最大长度 self.max_token_per_sent
        '''
        if len(origin_token_seq) > self.max_token_per_sent:
            return origin_token_seq[:self.max_token_per_sent]
        else:
            return origin_token_seq + [0 for _ in range(self.max_token_per_sent-len(origin_token_seq))]

    def tokenize(self, path, test_mode=False):
        '''
        处理指定的数据集分割，处理后每条数据中的 sentence 都将转化成对应的 token 序列。
        '''
        idss = []
        labels = []
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                one_data = json.loads(line)  # 读取一条数据
                sent = one_data['sentence']
                #-----------------------------------------------------begin-----------------------------------------------------#
                # 若要采用预训练的 embedding, 需在此处对 sent 进行分词
                words = list(jieba.cut(sent))  # 使用jieba进行中文分词
                # 过滤停用词
                words = [w for w in words if w.strip() and w not in self.stopwords]

                #------------------------------------------------------end------------------------------------------------------#
                # 向词典中添加词
                for word in words:
                    self.dictionary.add_word(word)

                ids = []
                for word in words:
                    ids.append(self.dictionary.word2tkn.get(word, 1))  # 未知词使用[UNK]的id=1
                idss.append(self.pad(ids))
                
                # 测试集无标签，在 label 中存测试数据的 id，便于最终预测文件的打印
                if test_mode:
                    label = json.loads(line)['id']      
                    labels.append(label)
                else:
                    label = json.loads(line)['label']
                    labels.append(self.dictionary.label2idx[label])

            idss = torch.tensor(np.array(idss))
            labels = torch.tensor(np.array(labels)).long()
            
        return TensorDataset(idss, labels)