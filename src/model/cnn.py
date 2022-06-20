import torch
import torch.nn as nn
import torch.nn.functional as F

class Cnn(nn.Module):
    def __init__(self, n_word, hidden_size):
        super(Cnn, self).__init__()
        self.out_size = hidden_size
        self.conv1 = nn.Conv2d(1, self.out_size, (4, 50))
        self.conv2 = nn.Conv2d(1, self.out_size, (3, 50))
        self.conv3 = nn.Conv2d(1, self.out_size, (2, 50))

        self.pool1 = nn.MaxPool1d(n_word - 3)
        self.pool2 = nn.MaxPool1d(n_word - 2)
        self.pool3 = nn.MaxPool1d(n_word - 1)

        self.softmax = nn.Softmax(1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(3 * self.out_size, 2)

    def forward(self, input):
        # 增加维度
        input = input.unsqueeze(1)

        out1 = F.relu(self.conv1(input)).squeeze(3)
        out2 = F.relu(self.conv2(input)).squeeze(3)
        out3 = F.relu(self.conv3(input)).squeeze(3)

        out1 = self.pool1(out1).squeeze(2)
        out2 = self.pool2(out2).squeeze(2)
        out3 = self.pool3(out3).squeeze(2)
        
        out = torch.cat((out1, out2, out3), 1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.softmax(out)
        return out
