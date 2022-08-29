from bert_base import BertBaseModel
from deberta_base import DebertaBaseModel

dc_text = "Making choices in life can be very difficult. People often ask for advice when they can not decide on one thing. It's always good to ask others for their advice when making a choice. When you have multiple opinions you have the ability to make the best choice for yourself."
dc_type = "Lead"
essay = open('./dc_eval_models/sample_essay.txt', 'r').read()

bert_base = BertBaseModel('./dc_eval_models/pretrained/bert-base.pth')
bert_res = bert_base.predict(dc_text, dc_type, essay)
print('BERT result:', bert_res)

deberta_base = DebertaBaseModel([
    f'./dc_eval_models/pretrained/deberta/fold_{i}.pth' for i in range(5)
    ], './dc_eval_models/pretrained/deberta/tokenizer')
deberta_res = deberta_base.predict(dc_text, dc_type, essay)
print('DeBERTa result:', deberta_res)