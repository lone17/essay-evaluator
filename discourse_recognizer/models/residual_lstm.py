import torch.nn as nn
import torch.nn.functional as F


class ResidualLSTM(nn.Module):

    def __init__(self, d_model, rnn):
        super(ResidualLSTM, self).__init__()
        self.downsample = nn.Linear(d_model, d_model // 2)
        if rnn == 'GRU':
            self.LSTM = nn.GRU(d_model // 2, d_model // 2, num_layers=2, bidirectional=False, dropout=0.2)
        else:
            self.LSTM = nn.LSTM(d_model // 2, d_model // 2, num_layers=2, bidirectional=False, dropout=0.2)
        self.dropout1 = nn.Dropout(0.2)
        self.norm1 = nn.LayerNorm(d_model // 2)
        self.linear1 = nn.Linear(d_model // 2, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout2 = nn.Dropout(0.2)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        res = x
        x = self.downsample(x)
        x, _ = self.LSTM(x)
        x = self.dropout1(x)
        x = self.norm1(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.dropout2(x)
        x = res + x
        return self.norm2(x)
