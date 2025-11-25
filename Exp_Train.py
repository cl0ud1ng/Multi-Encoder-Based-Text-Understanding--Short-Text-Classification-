import torch
import torch.nn as nn
import time
import json
import os
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import  DataLoader
from Exp_DataSet import Corpus
from Exp_Model import BiLSTM_model, Transformer_model


def train():
    '''
    进行训练
    '''
    max_valid_acc = 0
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'valid_acc': []
    }
    
    for epoch in range(num_epochs):
        model.train()

        total_loss = []
        total_true = []

        tqdm_iterator = tqdm(data_loader_train, dynamic_ncols=True, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for data in tqdm_iterator:
            # 选取对应批次数据的输入和标签
            batch_x, batch_y = data[0].to(device), data[1].to(device)

            # 模型预测
            y_hat = model(batch_x)

            loss = loss_function(y_hat, batch_y)

            optimizer.zero_grad()   # 梯度清零
            loss.backward()         # 计算梯度
            optimizer.step()        # 更新参数

            y_hat = torch.tensor([torch.argmax(_) for _ in y_hat]).to(device)
            
            total_true.append(torch.sum(y_hat == batch_y).item())
            total_loss.append(loss.item())

            tqdm_iterator.set_postfix(loss=sum(total_loss) / len(total_loss),
                                      acc=sum(total_true) / (batch_size * len(total_true)))
        
        tqdm_iterator.close()

        train_loss = sum(total_loss) / len(total_loss)
        train_acc = sum(total_true) / (batch_size * len(total_true))

        valid_acc = valid()

        if valid_acc > max_valid_acc:
            max_valid_acc = valid_acc
            torch.save(model, os.path.join(output_folder, "model.ckpt"))
        
        # 根据验证集准确率调整学习率
        scheduler.step(valid_acc)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(valid_acc)

        print(f"epoch: {epoch}, train loss: {train_loss:.4f}, train accuracy: {train_acc*100:.2f}%, valid accuracy: {valid_acc*100:.2f}%")
    
    return history


def plot_history(history):
    '''
    绘制训练历史曲线
    '''
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 绘制 Loss 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'loss_curve.png'), dpi=150)
    plt.show()
    
    # 绘制 Accuracy 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [acc * 100 for acc in history['train_acc']], 'b-', label='Train Accuracy')
    plt.plot(epochs, [acc * 100 for acc in history['valid_acc']], 'r-', label='Valid Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'accuracy_curve.png'), dpi=150)
    plt.show()
    
    print(f"Plots saved to {output_folder}")


def valid():
    '''
    进行验证，返回模型在验证集上的 accuracy
    '''
    total_true = []

    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader_valid, dynamic_ncols=True):
            batch_x, batch_y = data[0].to(device), data[1].to(device)

            y_hat = model(batch_x)
            # 取分类概率最大的类别作为预测的类别
            y_hat = torch.tensor([torch.argmax(_) for _ in y_hat]).to(device)

            total_true.append(torch.sum(y_hat == batch_y).item())

        return sum(total_true) / (batch_size * len(total_true))


def predict():
    '''
    读取训练好的模型对测试集进行预测，并生成结果文件
    '''
    test_ids = [] 
    test_pred = []

    model = torch.load(os.path.join(output_folder, "model.ckpt"), weights_only=False).to(device)
    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader_test, dynamic_ncols=True): 
            batch_x, batch_y = data[0].to(device), data[1]

            y_hat = model(batch_x)
            y_hat = torch.tensor([torch.argmax(_) for _ in y_hat])

            test_ids += batch_y.tolist()
            test_pred += y_hat.tolist()

    # 写入文件
    with open(os.path.join(output_folder, "predict.json"), "w") as f:
        for idx, label_idx in enumerate(test_pred):
            one_data = {}
            one_data["id"] = test_ids[idx]
            one_data["pred_label_desc"] = dataset.dictionary.idx2label[label_idx][1]
            json_data = json.dumps(one_data)    # 将字典转为json格式的字符串
            f.write(json_data + "\n")
            

if __name__ == '__main__':
    dataset_folder = './data/tnews_public'
    output_folder = './output'
    embedding_path = './sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'  # 预训练词向量路径
    preprocessed_path = './output/preprocessed_data.pkl'  # 预处理数据缓存路径

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    #-----------------------------------------------------begin-----------------------------------------------------#
    # 以下为超参数，可根据需要修改
    embedding_dim = 300     # 每个词向量的维度（与预训练词向量维度匹配）
    max_token_per_sent = 50 # 每个句子预设的最大 token 数
    batch_size = 32
    num_epochs = 12         # 增加训练轮数，配合学习率衰减 12 for BiLstm; 20 for transformer
    lr = 5e-3               # 初始学习率 5e-3 for BiLstm; 1e-3 for transformer
    #------------------------------------------------------end------------------------------------------------------#

    # 检查是否存在预处理缓存，若存在则直接加载，否则重新处理并保存
    os.makedirs(output_folder, exist_ok=True)
    if os.path.exists(preprocessed_path):
        dataset = Corpus.load_preprocessed(preprocessed_path)
        embedding_weight = dataset.embedding_weight
    else:
        dataset = Corpus(dataset_folder, max_token_per_sent)
        # 加载预训练词向量
        embedding_weight = dataset.load_pretrained_embedding(embedding_path, embedding_dim)
        # 保存预处理数据
        dataset.save_preprocessed(preprocessed_path)

    vocab_size = len(dataset.dictionary.tkn2word)
    num_classes = len(dataset.dictionary.idx2label)  # 获取类别数量

    data_loader_train = DataLoader(dataset=dataset.train, batch_size=batch_size, shuffle=True)
    data_loader_valid = DataLoader(dataset=dataset.valid, batch_size=batch_size, shuffle=False)
    data_loader_test = DataLoader(dataset=dataset.test, batch_size=batch_size, shuffle=False)

    #-----------------------------------------------------begin-----------------------------------------------------#
    # 可修改选择的模型以及传入的参数
    # 方案1: BiLSTM模型
    model = BiLSTM_model(vocab_size=vocab_size, ntoken=max_token_per_sent, d_emb=embedding_dim, d_hid=128, num_classes=num_classes, embedding_weight=embedding_weight).to(device)
    
    # 方案2: Transformer模型
    #model = Transformer_model(vocab_size=vocab_size, ntoken=max_token_per_sent, d_emb=embedding_dim, nhead=6, d_hid=1024, nlayers=4, num_classes=num_classes, embedding_weight=embedding_weight).to(device)
    #------------------------------------------------------end------------------------------------------------------#
    
    # 设置损失函数
    loss_function = nn.CrossEntropyLoss()
    # 设置优化器                                       
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    # 学习率调度器：验证集性能不提升时降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    # 进行训练
    history = train()
    
    # 绘制训练曲线
    plot_history(history)

    # 对测试集进行预测
    predict()
