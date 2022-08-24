from dis import dis
from typing import List
from discourse_evaluator import DiscourseEvaluator

from discourse_recognizer import DiscourseRecognizer
from discourse_classifier import DiscourseClassifier


class EssayEvaluator:
    def __init__(
        self, 
        discourse_recognizer: DiscourseRecognizer,
        discourse_classifer: DiscourseClassifier, 
        discourse_evaluator: DiscourseEvaluator, 
        *args, **kwargs) -> None:

        self.recognizer = discourse_recognizer
        self.classifier = discourse_classifer
        self.evaluator = discourse_evaluator
    
    def process(self, essay: str, *args, **kwargs) -> List[dict]:
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
        discourses = self.recognizer.process(essay, args, kwargs)
        for i in range(len(discourses)):
            st = discourses[i]["start"]
            ed = discourses[i]["end"]
            dc_txt = essay[st:ed + 1]

            eval = self.evaluator.process(dc_txt, args, kwargs)
            discourses[i] = discourses[i] | eval

        return discourses