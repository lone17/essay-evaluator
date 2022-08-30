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

def html_create():
    # Creating the HTML file
    file_html = open("demo.html", "w")
    # Adding the input data to the HTML file
    file_html.write("""<html>
    <head>
    <title>HTML File</title>
    </head> 
    <body>
    <img src="banner_demoday.jpg" alt="demo banner">
    </body>
    </html>""")
    # Saving the data into the HTML file
    file_html.close()
    
demo = gr.Interface(
    
    fn=greet,
    inputs=gr.Textbox(lines=2,
        label = "Essay",
        placeholder="Please input your essay here to evaluate..."),
    outputs= gr.Highlightedtext(
      label = "Segmentation",
      combine_adjacent = True,
    ),
    examples=[["""Do you really thing a land form could make a face ? To me that seems something is very unlikely. A planet that we do not look alike can make a face as a organism on earth? Well i don't think so and here is why.

The photo isn't clearly shown as a face it just looks like a planets land form.

It may aslo be a feture that is only unique to this palnet like the moon has thousands of craders. I just feel as if Earth has all the water and green grass on it mayb Mars has these land forms instead. Also it might be like how people gaze at the stars or the clouds to see if they make an object or a human face. If you look closley you can see that the

"face" has ridges or canyon. So maybe the ridegs/canyons made the land form that way.

NASA on the other hand and says there is likelyness of a human face. They also believe it is an alien artifac. According to the picture they say because of the shadows it portraies eyes, nose and mouth. They say that in the years that they took the picture it has developed more and more over the spands of years.

In conclusion the "face'' to me is still just a land form. Just because it looks as if it is something could not just be made out of a human face. If they had more evidence then it would be believeable but until then my opnion is still that it is a land form.                   """],[
    """Well, me personally, i dont think its a good idea for a senator to change to that. I feel ike everyone should remain voting as usaul. If not this wouldnt feel like america anymore; America is all about freedom and the power of free speech....etc. This is what makes this country great; no dictator telling us what to do everyday. Without the power of voting this country would not feel the same. It wouldnt feel like america anymore.

Also, althoughs theres a lot of cons to this, really its just more worth it. It makes people feel impowered. Knowning that they're making an imoportant decision in life. It makes people feel free and glad that they live in america. Where you have a choice that yours and freedom at the palm of your hands. Plus, sometimes people have more different opinions than others on voting. You just cant count on some people that you dont know to choose for you.

The popular voting process doesnt seem bad at all but i dont think some people would enjoy that very much so.    """]],
description=(
            "<div>"
            "<img  src='file/banner_demoday2.jpg' alt='image One'>"
            + "</div>"
            ),
)
demo.launch(share=True)