from typing import List

from discourse_recognizer import DiscourseRecognizer
from discourse_classifier import DiscourseClassifier


class EssayEvaluator:
    def __init__(self, discourse_recognizer,  discourse_classifer, discourse_evaluator, 
                 *args, **kwargs) -> None:
        pass
    
    def process(essay: str, *args, **kwargs) -> List[dict]:
        """_summary_

        Args:
            essay (str): _description_

        Returns:
            _type_: _description_
        
        
        Returns a list of dict, each dict represents a discourse text
        [
            {
                "start": int, -> the start position (character index)
                "end": int, -> the end position (character index)
                "type": str, -> discourse type
                "effectiveness": str, -> discourse effectiveness
                "score": float -> effectiveness score
            },
            ...
        ]
        """
        pass
        