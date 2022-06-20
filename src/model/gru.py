import torch
import torch.nn as nn
import torch.nn.functional as F

class Gru(nn.Module):
    def __init__(self, hid_size):
        super(Gru, self).__init__()
        self.rnn = nn.GRU(
            input_size = 50,
            hidden_size = hid_size,
            num_layers = 1,
            batch_first = True,
        )
        self.out = nn.Linear(hid_size, 2)
        self.softmax = nn.Softmax(1)

    def forward(self, input):
        r_out, _ = self.rnn(input, None)
        output = self.out(r_out[:, -1, :])
        output = self.softmax(output)
        return output