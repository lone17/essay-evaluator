import gradio as gr

from essay_evaluator import EssayEvaluator
import gradio as gr
"""TODO
1. Load the model
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
2. Create the UI
3. Serve
""" 

evaluator = EssayEvaluator(
    discourse_recognizer_params={},
    discourse_evaluator_params={"weights": "./discourse_evaluator/pretrained/bert-base.pth"}
)

def splitter(start, end, str):
    pieces = str.split()
    return " ".join(pieces[start : end + 1])

def greet(essay):
    essay_list = evaluator.process(essay)
    essay_list.append({"start" : 0, "end": 0, "type": None, "effectiveness": None, "score": None})
    sorted_essay = sorted(essay_list, key=lambda d: d['start'])
    outputText = []
    for i in range(1, len(sorted_essay)):
        if (sorted_essay[i]['start'] > sorted_essay[i - 1]['end'] + 1):
            segmentText = splitter(sorted_essay[i - 1]['end'] + 1, sorted_essay[i]['start'] - 1, essay) + '\n'
            outputText.append((segmentText, None))
        segmentText = f"[TYPE: {sorted_essay[i]['type']}, COMMENT: {sorted_essay[i]['effectiveness']}, SCORE: {sorted_essay[i]['score']}]\n"
        outputText.append((segmentText, "Evaluate"))
        segmentText = splitter(sorted_essay[i]['start'], sorted_essay[i]['end'], essay) + '\n'
        outputText.append((segmentText, sorted_essay[i]['type']))
        outputText.append(('\n', None))
    return outputText


demo = gr.Interface(
    
    fn=greet,
    inputs=gr.Textbox(lines=2, placeholder="Please input your essay here to evaluate..."),
    outputs= gr.Highlightedtext(
      label = "Segmentation",
      combine_adjacent = True,
    )
    # .style(color_map={"Evaluate": "green"}),
)
demo.launch(share=True)