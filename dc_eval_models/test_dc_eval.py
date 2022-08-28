from bert_base import BertBaseModel
#from deberta_base import DebertaBaseModel

dc_text = "Making choices in life can be very difficult. People often ask for advice when they can not decide on one thing. It's always good to ask others for their advice when making a choice. When you have multiple opinions you have the ability to make the best choice for yourself."
dc_type = "Lead"
essay = open('./dc_eval_models/sample_essay.txt', 'r').read()

bert_base = BertBaseModel()
#deberta_base = DebertaBaseModel()

bert_res = bert_base.predict(dc_text, dc_type, essay)
#deberta_res = deberta_base.predict(dc_text, dc_type, essay)

print('BERT result:', bert_res)
#print('DeBERTa result:', deberta_res)