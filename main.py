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

demoText = "Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean commodo ligula eget dolor. Aenean massa. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Donec quam felis, ultricies nec, pellentesque eu, pretium quis, sem. Nulla consequat massa quis enim. Donec pede justo, fringilla vel, aliquet nec, vulputate eget, arcu. In enim justo, rhoncus ut, imperdiet a, venenatis vitae, justo. Nullam dictum felis eu pede mollis pretium. Integer tincidunt. Cras dapibus. Vivamus elementum semper nisi. Aenean vulputate eleifend tellus. Aenean leo ligula, porttitor eu, consequat vitae, eleifend ac, enim. Aliquam lorem ante, dapibus in, viverra quis, feugiat a, tellus. Phasellus viverra nulla ut metus varius laoreet. Quisque rutrum. Aenean imperdiet. Etiam ultricies nisi vel augue. Curabitur ullamcorper ultricies nisi. Nam eget dui. Etiam rhoncus. Maecenas tempus, tellus eget condimentum rhoncus, sem quam semper libero, sit amet adipiscing sem neque sed ipsum. Nam quam nunc, blandit vel, luctus pulvinar, hendrerit id, lorem. Maecenas nec odio et ante tincidunt tempus. Donec vitae sapien ut libero venenatis faucibus. Nullam quis ante. Etiam sit amet orci eget eros faucibus tincidunt. Duis leo. Sed fringilla mauris sit amet nibh. Donec sodales sagittis magna. Sed consequat, leo eget bibendum sodales, augue velit cursus nunc, quis gravida magna mi a libero. Fusce vulputate eleifend sapien. Vestibulum purus quam, scelerisque ut, mollis sed, nonummy id, metus. Nullam accumsan lorem in dui. Cras ultricies mi eu turpis hendrerit fringilla. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; In ac dui quis mi consectetuer lacinia. Nam pretium turpis et arcu. Duis arcu tortor, suscipit eget, imperdiet nec, imperdiet iaculis, ipsum. Sed aliquam ultrices mauris. Integer ante arcu, accumsan a, consectetuer eget, posuere ut, mauris. Praesent adipiscing. Phasellus ullamcorper ipsum rutrum nunc. Nunc nonummy metus. Vestibulum volutpat pretium libero. Cras id dui. Aenean ut eros et nisl sagittis vestibulum. Nullam nulla eros, ultricies sit amet, nonummy id, imperdiet feugiat, pede. Sed lectus. Donec mollis hendrerit risus. Phasellus nec sem in justo pellentesque facilisis. Etiam imperdiet imperdiet orci. Nunc nec neque. Phasellus leo dolor, tempus non, auctor et, hendrerit quis, nisi. Curabitur ligula sapien, tincidunt non, euismod vitae, posuere imperdiet, leo. Maecenas malesuada. Praesent congue erat at massa. Sed cursus turpis vitae tortor. Donec posuere vulputate arcu. Phasellus accumsan cursus velit. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Sed aliquam, nisi quis porttitor congue, elit erat euismod orci, ac placerat dolor lectus quis orci. Phasellus consectetuer vestibulum elit. Aenean tellus metus, bibendum sed, posuere ac, mattis non, nunc. Vestibulum fringilla pede sit amet augue. In turpis. Pellentesque posuere. Praesent turpis. Aenean posuere, tortor sed cursus feugiat, nunc augue blandit nunc, eu sollicitudin urna dolor sagittis lacus. Donec elit libero, sodales nec, volutpat a, suscipit non, turpis. Nullam sagittis. Suspendisse pulvinar, augue ac venenatis condimentum, sem libero volutpat nibh, nec pellentesque velit pede quis nunc. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Fusce id purus. Ut varius tincidunt libero. Phasellus dolor. Maecenas vestibulum mollis"

def splitter(start, end, str):
    pieces = str.split()
    return " ".join(pieces[start : end + 1])

def greet(essay):
    essay_list = EssayEvaluator.process(essay)
    #essay = demoText
    #essay_list = [{"start" : 1, "end": 50, "type": "Lead", "effectiveness": "Useless", "score": 80},
    #           {"start": 106, "end": 200, "type": "Position", "effectiveness": "Useful", "score":60},
    #            {"start": 52, "end": 58, "type": "Position", "effectiveness": "Useful", "score":60},
    #            {"start": 58, "end": 70, "type": "Position", "effectiveness": "Useful", "score":60},
    #             ]
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
demo.launch()