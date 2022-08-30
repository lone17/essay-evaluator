import torch.nn as nn
from residual_lstm import ResidualLSTM


class ConvLSTMHead(nn.Module):
    def __init__(self):
        super(ConvLSTMHead, self).__init__()
        self.downsample = nn.Sequential(nn.Linear(1024, 256))
        self.conv1 = nn.Sequential(nn.Conv1d(256, 256, 3, padding=1),
                                   nn.ReLU())
        self.norm1 = nn.LayerNorm(256)
        self.conv2 = nn.Sequential(nn.Conv1d(256, 256, 3, padding=1),
                                   nn.ReLU())
        self.norm2 = nn.LayerNorm(256)
        # self.lstm=nn.LSTM(256,256,2,bidirectional=True)
        self.lstm = ResidualLSTM(256)
        self.upsample = nn.Sequential(nn.Linear(256, 1024), nn.ReLU())
        self.classification_head = nn.Sequential(nn.Linear(1024, 15))

    def forward(self, x):
        x = self.downsample(x)
        res = x
        x = self.conv1(x.permute(0, 2, 1))
        x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.conv2(x)
        x = self.norm1(x.permute(0, 2, 1))
        x = x + res
        x = self.lstm(x.permute(1, 0, 2))
        x = x.permute(1, 0, 2)
        x = self.upsample(x)
        x = self.classification_head(x)
        # print(x.shape)
        # exit()
        return x
