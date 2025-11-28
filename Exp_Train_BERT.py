"""
BERT 文本分类训练脚本
使用预训练的 BERT 模型进行文本分类，冻结预训练层，只训练分类层。
"""
import torch
import torch.nn as nn
import time
import json
import os
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from Exp_DataSet import BertCorpus
from Exp_Model import BERT_model


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
            # BERT 数据格式: input_ids, attention_mask, token_type_ids, labels
            input_ids = data[0].to(device)
            attention_mask = data[1].to(device)
            token_type_ids = data[2].to(device)
            labels = data[3].to(device)

            # 模型预测
            y_hat = model(input_ids, attention_mask, token_type_ids)

            loss = loss_function(y_hat, labels)

            optimizer.zero_grad()   # 梯度清零
            loss.backward()         # 计算梯度
            optimizer.step()        # 更新参数

            y_pred = torch.argmax(y_hat, dim=1)
            
            total_true.append(torch.sum(y_pred == labels).item())
            total_loss.append(loss.item())

            tqdm_iterator.set_postfix(loss=sum(total_loss) / len(total_loss),
                                      acc=sum(total_true) / (batch_size * len(total_true)))
        
        tqdm_iterator.close()

        train_loss = sum(total_loss) / len(total_loss)
        train_acc = sum(total_true) / (batch_size * len(total_true))

        valid_acc = valid()

        if valid_acc > max_valid_acc:
            max_valid_acc = valid_acc
            torch.save(model, os.path.join(output_folder, "bert_model.ckpt"))
        
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
    plt.title('BERT Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'bert_loss_curve.png'), dpi=150)
    plt.show()
    
    # 绘制 Accuracy 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [acc * 100 for acc in history['train_acc']], 'b-', label='Train Accuracy')
    plt.plot(epochs, [acc * 100 for acc in history['valid_acc']], 'r-', label='Valid Accuracy')
    plt.title('BERT Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'bert_accuracy_curve.png'), dpi=150)
    plt.show()
    
    print(f"Plots saved to {output_folder}")


def valid():
    '''
    进行验证，返回模型在验证集上的 accuracy
    '''
    total_true = []
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader_valid, dynamic_ncols=True, desc='Validating'):
            input_ids = data[0].to(device)
            attention_mask = data[1].to(device)
            token_type_ids = data[2].to(device)
            labels = data[3].to(device)

            y_hat = model(input_ids, attention_mask, token_type_ids)
            y_pred = torch.argmax(y_hat, dim=1)

            total_true.append(torch.sum(y_pred == labels).item())
            total_samples += labels.size(0)

    return sum(total_true) / total_samples


def predict():
    '''
    读取训练好的模型对测试集进行预测，并生成结果文件
    '''
    test_ids = [] 
    test_pred = []

    model = torch.load(os.path.join(output_folder, "bert_model.ckpt"), weights_only=False).to(device)
    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader_test, dynamic_ncols=True, desc='Predicting'): 
            input_ids = data[0].to(device)
            attention_mask = data[1].to(device)
            token_type_ids = data[2].to(device)
            batch_ids = data[3]  # 测试集的 id

            y_hat = model(input_ids, attention_mask, token_type_ids)
            y_pred = torch.argmax(y_hat, dim=1)

            # batch_ids 可能是列表或 tensor
            if isinstance(batch_ids, torch.Tensor):
                test_ids += batch_ids.tolist()
            else:
                test_ids += list(batch_ids)
            test_pred += y_pred.cpu().tolist()

    # 写入文件
    with open(os.path.join(output_folder, "bert_predict.json"), "w", encoding='utf-8') as f:
        for idx, label_idx in enumerate(test_pred):
            one_data = {}
            one_data["id"] = test_ids[idx]
            one_data["pred_label_desc"] = dataset.idx2label[label_idx][1]
            json_data = json.dumps(one_data, ensure_ascii=False)
            f.write(json_data + "\n")
    
    print(f"Predictions saved to {os.path.join(output_folder, 'bert_predict.json')}")


if __name__ == '__main__':
    dataset_folder = './data/tnews_public'
    output_folder = './output'

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    #-----------------------------------------------------begin-----------------------------------------------------#
    # BERT 超参数配置
    bert_model_name = 'bert-base-chinese'  # 轻量级预训练模型（4层，快3倍）
    max_length = 128            # BERT 输入的最大长度
    batch_size = 16             # BERT 模型较大，使用较小的 batch size
    num_epochs = 5              # 冻结预训练层时，通常不需要太多 epoch
    lr = 2e-4                   # 分类层学习率（预训练层冻结，可以用较大学习率）
    freeze_bert = True          # 是否冻结 BERT 预训练层
    #------------------------------------------------------end------------------------------------------------------#

    os.makedirs(output_folder, exist_ok=True)
    
    # 预处理数据缓存路径
    bert_preprocessed_path = './output/bert_preprocessed_data.pkl'

    # 检查是否存在预处理缓存，若存在则直接加载，否则重新处理并保存
    if os.path.exists(bert_preprocessed_path):
        dataset = BertCorpus.load_preprocessed(bert_preprocessed_path, bert_model_name=bert_model_name)
    else:
        print("Loading BERT dataset...")
        dataset = BertCorpus(dataset_folder, bert_model_name=bert_model_name, max_length=max_length)
        # 保存预处理数据
        dataset.save_preprocessed(bert_preprocessed_path)

    num_classes = len(dataset.idx2label)
    print(f"Number of classes: {num_classes}")

    data_loader_train = DataLoader(dataset=dataset.train, batch_size=batch_size, shuffle=True)
    data_loader_valid = DataLoader(dataset=dataset.valid, batch_size=batch_size, shuffle=False)
    data_loader_test = DataLoader(dataset=dataset.test, batch_size=batch_size, shuffle=False)

    #-----------------------------------------------------begin-----------------------------------------------------#
    # 初始化 BERT 分类模型
    model = BERT_model(
        bert_model_name=bert_model_name,
        num_classes=num_classes,
        dropout=0.1,
        freeze_bert=freeze_bert
    ).to(device)
    
    # 打印可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    #------------------------------------------------------end------------------------------------------------------#
    
    # 设置损失函数
    loss_function = nn.CrossEntropyLoss()
    
    # 设置优化器（只优化需要梯度的参数）
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)

    # 进行训练
    print("\n" + "="*50)
    print("Starting BERT training (frozen pretrained layers)")
    print("="*50 + "\n")
    
    history = train()
    
    # 绘制训练曲线
    plot_history(history)

    # 对测试集进行预测
    predict()
