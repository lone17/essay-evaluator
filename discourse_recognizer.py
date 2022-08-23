from typing import List

from essay_segmentor import EssaySegmentor
from discourse_classifier import DiscourseClassifier


class DiscourseRecognizer:
    def __init__(self, essay_segmentor, discourse_classifer) -> None:
        pass
    
    def process(self, essay: str, *args, **kwargs) -> List[dict]:
        """_summary_

        Args:
            essay (str): _description_

        Returns:
            List[dict]: _description_
        
        Returns a list of dict, each dict represents a discourse text
        [
            {
                "start": int, -> the start position (character index)
                "end": int, -> the end position (character index)
                "type": str, -> discourse type
            },
            ...
        ]
        """        
        pass
        
        