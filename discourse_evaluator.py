from dc_eval_models.bert_base import BertBaseModel
from dc_eval_models.dc_eval_base import DiscourseEvalBaseModel

class DiscourseEvaluator:
    def __init__(self, model: DiscourseEvalBaseModel) -> None:
        self.model = model # can replace with DeBERTa
    
    def process(self, discourse: str, discourse_type: str, essay: str, *args, **kwargs) -> dict:
        """_summary_

        Args:
            discourse (str): _description_

        Returns:
            dict: _description_
        
        {
            "effectiveness": str -> discourse effectiveness
            "score": float -> effectiveness score
        }
        """
        return self.model.predict(discourse, discourse_type, essay)