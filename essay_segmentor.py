from typing import List


class EssaySegmentor:
    def __init__(self, weights, *args, **kwargs) -> None:
        pass
    
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
            },
            ...
        ]
        """
        pass