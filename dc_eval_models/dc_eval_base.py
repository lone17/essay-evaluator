import torch, numpy as np

class DiscourseEvalBaseModel:
    def __init__(self):
        self.target_list = ['Ineffective', 'Adequate', 'Effective']
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z))
        
    def predict(self, dc_text: str, dc_type: str, essay: str) -> dict:
        pass