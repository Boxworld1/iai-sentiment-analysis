import torch
import torch.nn as nn
import torch.nn.functional as F

class Mlp(nn.Module):
    def __init__(self, sentence_size, hid_size):
        super(Mlp, self).__init__()
        self.hidden_size = hid_size
        self.sentence_size = sentence_size
        self.fc1 = nn.Linear(50, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size * self.sentence_size, 2)
        self.softmax = nn.Softmax(1)

    def forward(self, input):
        output = self.fc1(input)
        output = F.relu(output)
        output = output.view(output.size(0), -1)
        output = self.fc2(output)
        output = self.softmax(output)
        return output
