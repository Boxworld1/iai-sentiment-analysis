import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import gensim
import os

class DataSet(Dataset):
    def __init__(self, filepath, filename, sentence_size):
        # 要开启的文件名
        self.filepath = os.path.join(filepath, filename)
        inputFile = open(self.filepath, 'r')
        dataset = inputFile.readlines()

        # 数据初始化
        self.length = len(dataset)
        self.data = []
        self.label = []

        # 加载词向量
        model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(filepath, 'wiki_word2vec_50.bin'), binary=True)

        # 遍历行中元素
        for str in dataset:
            # 以空格分开
            sentence = str.split()
            label = sentence[0]
            del sentence[0]

            # 词向量化
            wordlist = []
            for word in sentence:
                if len(wordlist) < sentence_size:
                    if word in model:
                        wordlist.append(model[word])
                    else:
                        wordlist.append(np.random.rand(50))

            # 句子长度补足
            while len(wordlist) < sentence_size:
                wordlist.append(np.random.rand(50))

            # 放入数据库中
            self.data.append(wordlist)
            self.label.append(label)
        

    def __getitem__(self, index):
        sentence = torch.tensor(np.array(self.data[index]))
        label = torch.tensor(int(self.label[index]))
        return sentence, label

    def __len__(self):
        return self.length