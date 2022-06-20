import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import gensim
import os
import argparse

from src.model.cnn import Cnn
from src.model.rnn import Rnn
from src.model.gru import Gru
from src.model.lstm import Lstm
from src.model.mlp import Mlp
from src.data.dataset import DataSet

# Hyper Parameters
BATCH_SIZE = 64
SENTENCE_SIZE = 60
HIDDEN_SIZE = 64
EPOCH = 100
LR = 0.001
GPUID = 5

def train(model, train_loader, optimizer, loss_func):
    train_total, train_loss =  0., 0.
    train_tp, train_fp, train_fn, train_tn = 0., 0., 0., 0.
    train_pre, train_rec = 0.2, 0.2

    # 训练模型
    model.train()
    for idx, (sentence, target) in enumerate(train_loader):
        sentence = sentence.to(device)
        target = target.to(device)

        # 前向传播
        output = model(sentence.float())
        # 计算损失
        loss = loss_func(output, target)
        train_total += sentence.size(0)
        # 后向传递
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 计算准确率
        train_loss += loss.item() * sentence.size(0)
        pred = torch.max(output, 1)[1].data.squeeze()
        for idx in range(len(pred)):
            if target[idx]:
                if pred[idx]:
                    train_tp += 1
                else:
                    train_fp += 1
            else:
                if pred[idx]:
                    train_fn += 1
                else:
                    train_tn += 1
    # 计算训练集指标
    train_acc = (train_tp + train_tn) / train_total
    train_loss = train_loss / train_total

    if train_tp:
        train_pre = train_tp / (train_tp + train_fp)
        train_rec = train_tp / (train_tp + train_fn)

    train_fm = 2.0 / (1.0 / train_pre + 1.0 / train_rec)
    return train_loss, train_acc, train_fm
    
def test(model, test_loader, loss_func):
    test_total, test_loss = 0., 0.
    test_tp, test_fp, test_fn, test_tn = 0., 0., 0., 0.
    test_pre, test_rec = 0.2, 0.2

    model.eval()
    for idx, (sentence, target) in enumerate(test_loader):
        sentence = sentence.to(device)
        target = target.to(device)

        output = model(sentence.float())
        loss = loss_func(output, target)
        
        # 计算准确率
        test_total += sentence.size(0)
        test_loss += loss.item() * sentence.size(0)
        pred = torch.max(output, 1)[1].data.squeeze()
        for idx in range(len(pred)):
            if target[idx]:
                if pred[idx]:
                    test_tp += 1
                else:
                    test_fp += 1
            else:
                if pred[idx]:
                    test_fn += 1
                else:
                    test_tn += 1

    # 计算测试集指标
    test_acc = (test_tp + test_tn) / test_total
    test_loss = test_loss / test_total

    if test_tp:
        test_pre = test_tp / (test_tp + test_fp)
        test_rec = test_tp / (test_tp + test_fn)

    test_fm = 2.0 / (1.0 / test_pre + 1.0 / test_rec)
    return test_loss, test_acc, test_fm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default='mlp', help="model name: mlp, cnn, rnn.")
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="batch size.")
    parser.add_argument("-hs", "--hidden_size", type=int, default=64, help="hidden layer size.")
    parser.add_argument("-sl", "--sentence_len", type=int, default=60, help="sentence max length.")
    parser.add_argument("-ep", "--epoch", type=int, default=100, help="epochs")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="learning rate.")
    parser.add_argument("-t", "--train", type=str, default='Y', help="train model. (Y/N)")
    parser.add_argument("-g", "--gpu_id", type=int, default=0, help="gpu id")
    args = parser.parse_args()

    model_name = args.model
    BATCH_SIZE = args.batch_size
    SENTENCE_SIZE = args.sentence_len
    HIDDEN_SIZE = args.hidden_size
    EPOCH = args.epoch
    LR = args.learning_rate
    GPUID = args.gpu_id
    
    path = os.getcwd()
    path = os.path.join(path, 'src', 'dataset')

    # 设备选择
    device_name = "cuda:" + str(GPUID)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    print('=============================')
    # 选择模型
    model = Rnn(HIDDEN_SIZE)
    if model_name == 'cnn':
        model = Cnn(SENTENCE_SIZE, HIDDEN_SIZE)
    # elif model_name == 'rnn':
    #     model = Rnn(HIDDEN_SIZE)
    elif model_name == 'lstm':
        model = Lstm(HIDDEN_SIZE)
    elif model_name == 'gru':
        model = Gru(HIDDEN_SIZE)
    else:
        model = Mlp(SENTENCE_SIZE, HIDDEN_SIZE)
        model_name = 'mlp'
        
    print('Model:         ', model_name)
    print('Batch size:    ', BATCH_SIZE)
    print('Epoch:         ', EPOCH)
    print('Hidden layer:  ', HIDDEN_SIZE)
    print('Sentence len:  ', SENTENCE_SIZE)
    print('Learning rate: ', LR)
    print('Device:        ', device)
    print('Train Status:  ', args.train)
    print('=============================')

    # 数据导入
    print('Data Unpacking...')
    train_set = DataSet(path, 'train.txt', SENTENCE_SIZE)
    train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True)
    test_set = DataSet(path, 'test.txt', SENTENCE_SIZE)
    test_loader = DataLoader(test_set, batch_size = BATCH_SIZE, shuffle = True)
    valid_set = DataSet(path, 'validation.txt', SENTENCE_SIZE)
    valid_loader = DataLoader(test_set, batch_size = BATCH_SIZE, shuffle = True)

    if not os.path.exists('result'):
        os.makedirs('result')
    if not os.path.exists('result/' + model_name):
        os.makedirs('result/' + model_name)

    cp_path = 'result/{}/{}_{}_{}_{}.pkl'.format(model_name, model_name, HIDDEN_SIZE, SENTENCE_SIZE, int(LR * 10000))

    model = model.float()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = LR)
    loss_func = nn.CrossEntropyLoss()

    if args.train == 'Y':
        print('Start training progress...')
        print('--------------------------------------------------------------------------------')
        print('Epoch | Tr. Loss | Tr. Acc. | Tr. F-Score | Test Loss | Test Acc. | Test F-Score')
        print('------+----------+----------+-------------+-----------+-----------+-------------')

        # 最优解 
        last_loss, best_acc, _ = test(model, test_loader, loss_func)
        cp = {
            'state_dict': model.state_dict(),
            'epoch': 0,
            'accuracy': best_acc,
        }
        torch.save(cp, cp_path)
        count = 0

        for epoch in range(EPOCH):
            train_loss, train_acc, train_fm = train(model, train_loader, optimizer, loss_func)
            test_loss, test_acc, test_fm = test(model, test_loader, loss_func)
            print('%5.0f' % (epoch + 1), '| %8.4f' % train_loss, '| %8.2f' % train_acc,
            '| %11.4f' % train_fm, '| %9.4f' % test_loss, '| %9.2f' % test_acc, '| %12.4f' % test_fm)

            # 若为当前最优
            if test_acc > best_acc:
                best_acc = test_acc
                cp = {
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'accuracy': test_acc,
                }
                torch.save(cp, cp_path)
            
            # 与前一个 epoch 相比，train_loss 差的绝对值小于 0.003 则
            if abs(train_loss - last_loss) < 0.003:
                count += 1
                # Early Stopping
                if count > 10:
                    break
            else:
                count = 0

            last_loss = train_loss

        print('--------------------------------------------------------------------------------')

    cp = torch.load(cp_path, map_location=device)
    model.load_state_dict(cp['state_dict'])
    _, acc, fs = test(model, valid_loader, loss_func)
    print('=============================')
    print('Result')
    print('Model:    {}'.format(model_name))
    print('Hidden:   {}'.format(HIDDEN_SIZE))
    print('Epoch:    {}'.format(cp['epoch']))
    print('Accuracy: {:.5f}'.format(acc))
    print('F-Score:  {:.5f}'.format(fs))
    print('=============================')