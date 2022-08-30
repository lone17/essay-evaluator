import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from .residual_lstm import ResidualLSTM

rearrange_indices = [14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


class TransformerModel(nn.Module):
    def __init__(self, DOWNLOADED_MODEL_PATH, rnn='LSTM'):
        super(TransformerModel, self).__init__()
        config_model = AutoConfig.from_pretrained(DOWNLOADED_MODEL_PATH + '/config.json')

        self.backbone = AutoModel.from_pretrained(
            DOWNLOADED_MODEL_PATH + '/pytorch_model.bin', config=config_model)

        self.lstm = ResidualLSTM(1024, rnn)
        self.classification_head = nn.Linear(1024, 15)

    def forward(self, x, attention_mask):
        x = self.backbone(input_ids=x, attention_mask=attention_mask, return_dict=False)[0]

        x = self.lstm(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.classification_head(x)

        return [x[:, :, rearrange_indices]]


class SlidingWindowTransformerModel(nn.Module):
    def __init__(self, DOWNLOADED_MODEL_PATH, rnn, window_size=512, edge_len=64):
        super(SlidingWindowTransformerModel, self).__init__()
        config_model = AutoConfig.from_pretrained(DOWNLOADED_MODEL_PATH + '/config.json')
        print(DOWNLOADED_MODEL_PATH)
        self.backbone = AutoModel.from_pretrained(
            DOWNLOADED_MODEL_PATH + '/pytorch_model.bin', config=config_model)

        self.lstm = ResidualLSTM(1024, rnn)
        self.classification_head = nn.Linear(1024, 15)
        self.window_size = window_size
        self.edge_len = edge_len
        self.inner_len = window_size - edge_len * 2

    def forward(self, input_ids, attention_mask):
        B, L = input_ids.shape

        if L <= self.window_size:
            x = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0]

        else:
            segments = (L - self.window_size) // self.inner_len
            if (L - self.window_size) % self.inner_len > self.edge_len:
                segments += 1
            elif segments == 0:
                segments += 1
            x = self.backbone(input_ids=input_ids[:, :self.window_size],
                              attention_mask=attention_mask[:, :self.window_size], return_dict=False)[0]
            for i in range(1, segments + 1):
                start = self.window_size - self.edge_len + (i - 1) * self.inner_len
                end = self.window_size - self.edge_len + (i - 1) * self.inner_len + self.window_size
                end = min(end, L)
                x_next = input_ids[:, start:end]
                mask_next = attention_mask[:, start:end]
                x_next = self.backbone(input_ids=x_next, attention_mask=mask_next, return_dict=False)[0]

                if i == segments:
                    x_next = x_next[:, self.edge_len:]
                else:
                    x_next = x_next[:, self.edge_len:self.edge_len + self.inner_len]
                x = torch.cat([x, x_next], 1)

        x = self.lstm(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.classification_head(x)

        return [x[:, :, rearrange_indices]]
