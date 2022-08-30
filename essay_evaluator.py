from typing import List

from discourse_recognizer import DiscourseRecognizer
from discourse_evaluator import DiscourseEvaluator

class EssayEvaluator:
    def __init__(
        self, 
        discourse_recognizer_params,
        discourse_evaluator_params, 
        *args, **kwargs) -> None:

        self.recognizer = DiscourseRecognizer(**discourse_recognizer_params)
        self.evaluator = DiscourseEvaluator(**discourse_evaluator_params)
    
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
        discourses = self.recognizer.process(essay)
        essay_split = essay.split()
        for i in range(len(discourses)):
            st = discourses[i]["start"]
            ed = discourses[i]["end"]

            dc_txt = " ".join(essay_split[st:ed + 1])
            dc_type = discourses[i]["type"]

            eval = self.evaluator.process(dc_txt, dc_type, essay, args, kwargs)
            discourses[i].update(eval)

        return discourses