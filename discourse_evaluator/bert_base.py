import torch, numpy as np
from transformers import BertModel, BertConfig, BertTokenizer

from .dc_eval_base import DiscourseEvalBaseModel

class FeedbackPrizeModel(torch.nn.Module):
    def __init__(self, target_size):
        super(FeedbackPrizeModel, self).__init__()
        self.bert_model = BertModel(BertConfig())
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, target_size)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert_model(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        output = self.dropout(output.pooler_output)
        output = self.linear(output)
        return output

class BertBaseModel(DiscourseEvalBaseModel):
    def __init__(self, weight: str):
        super(BertBaseModel, self).__init__()

        self.model = FeedbackPrizeModel(len(self.target_list))
        self.model.load_state_dict(torch.load(weight, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = 128

    def __prepare_input__(self, dc_text, dc_type, essay):
        text = ' '.join([dc_type, self.tokenizer.sep_token, dc_text, self.tokenizer.sep_token, essay])
        inputs = self.tokenizer.encode_plus(text.lower(),
                                            truncation=True,
                                            padding = 'max_length',
                                            add_special_tokens=True,
                                            return_attention_mask = True,
                                            return_token_type_ids= True,
                                            max_length=self.max_len,
                                            return_tensors = 'pt')

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids}

    def predict(self, dc_text: str, dc_type: str, essay: str) -> dict:
        input_data = self.__prepare_input__(dc_text, dc_type, essay)

        input_ids = input_data['input_ids'].to(self.device, dtype=torch.long)
        attention_mask = input_data['attention_mask'].to(self.device, dtype = torch.long)
        token_type_ids = input_data['token_type_ids'].to(self.device, dtype = torch.long)

        output = self.model.forward(input_ids, attention_mask, token_type_ids).flatten()
        output = output.detach().cpu().numpy()

        output = self.softmax(output)

        id = np.argmax(output)

        return {'effectiveness': self.target_list[id], 'score': output[id]}