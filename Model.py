import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class myRNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,  # RNN隐藏神经元个数
            num_layers=len(num_layers),  # RNN隐藏层个数
            batch_first=True
        )
        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, x, hidden):
        batch_size = x.shape[0]
        r_out, hidden = self.rnn(x, hidden)       
        r_out = r_out.view(-1, self.rnn.hidden_size)       
        output = self.fc(r_out)

        return output, hidden
