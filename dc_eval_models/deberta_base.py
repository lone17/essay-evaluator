from unicodedata import east_asian_width
import torch, codecs, torch.nn as nn, numpy as np
from text_unidecode import unidecode
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import Dict, List, Tuple

class CFG:
    CVs = []
    seed = 42
    lr = 3e-5
    epochs = 3
    n_fold = 5
    apex = True
    fast = True
    AMP = False
    n_splits = 5
    train = True
    wandb = False
    max_len = 512
    dropout = 0.1
    min_lr = 1e-6
    batch_size = 8
    freezing = True
    print_freq = 50
    target_size = 3
    num_workers = 0
    num_cycles = 0.5
    n_accumulate = 1
    scheduler = 'cosine'
    weigth_decay = 0.01
    num_warmup_steps = 0
    trn_fold = [0, 1, 2, 3, 4]
    gradient_checkpointing = True
    model = 'deberta-v3-base'
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=fast)

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div

def freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad = False

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9) #
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class FeedBackModel(nn.Module):
    def __init__(self, model_name):
        super(FeedBackModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        if CFG.gradient_checkpointing: (self.model).gradient_checkpointing_enable()
        if CFG.freezing:
            freeze((self.model).embeddings)
            freeze((self.model).encoder.layer[:2])
            CFG.after_freezed_parameters = filter(lambda parameter: parameter.requires_grad, (self.model).parameters())
        self.config = AutoConfig.from_pretrained(model_name)
        self.drop = nn.Dropout(p=CFG.dropout)
        self.pooler = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, CFG.target_size)
        
    def forward(self, ids, mask):
        out = self.model(input_ids = ids, attention_mask = mask, output_hidden_states = False)
        out = self.pooler(out.last_hidden_state, mask)
        out = self.drop(out)
        outputs = self.fc(out)
        return outputs.flatten()

def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]: return error.object[error.start : error.end].encode("utf-8"), error.end
def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]: return error.object[error.start : error.end].decode("cp1252"), error.end
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)

def resolve_encodings_and_normalize(text: str) -> str:
    text = (text.encode("raw_unicode_escape").decode("utf-8", errors = "replace_decoding_with_cp1252").encode("cp1252", errors = "replace_encoding_with_utf8").decode("utf-8", errors = "replace_decoding_with_cp1252"))
    text = unidecode(text)
    return text

class DebertaBaseModel:
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.target_list = ['Ineffective', 'Adequate', 'Effective']

    def __prepare_input__(self, dc_text: str, dc_type: str, essay: str):
        dc_text = resolve_encodings_and_normalize(dc_text)
        essay = resolve_encodings_and_normalize(essay)
        text = dc_type + ' ' + dc_text + '[SEP]' + essay

        inputs = CFG.tokenizer.encode_plus(text, truncation=True, add_special_tokens=True, max_length=CFG.max_len)
        samples = {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], }
        if 'token_type_ids' in inputs: samples['token_type_ids'] = inputs['token_type_ids']
        return samples

    def predict(self, dc_text: str, dc_type: str, essay: str) -> dict:
        data = self.__prepare_input__(dc_text, dc_type, essay)

        ids = data['input_ids'].to(self.device, dtype = torch.long)
        mask = data['attention_mask'].to(self.device, dtype = torch.long)

        predictions = []
        for i in CFG.trn_fold:
            model = FeedBackModel(CFG.model)
            model.load_state_dict(torch.load(f'./dc_eval_models/pretrained/deberta/fold{i}.pth', map_location=self.device))
            model.eval()
            model.to(self.device)

            pred = model.forward(ids, mask)
            pred = pred.detach().numpy()
            predictions.append(pred)

        output = np.mean(predictions, axis=1)
        id = np.argmax(output)

        return {'effectiveness': self.target_list[id], 'score': output[id]}