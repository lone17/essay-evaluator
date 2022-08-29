from .bert_base import BertBaseModel
from .deberta_base import DebertaBaseModel

class DiscourseEvaluator:
    __model_class = {
        'bert': BertBaseModel,
        'deberta': DebertaBaseModel
    }

    def __init__(self, weights, model_type='bert') -> None:
        _model = self.__model_class[model_type]
        self.model = _model(weights)
    
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