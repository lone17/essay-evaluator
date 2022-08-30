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


demo = gr.Interface(
    
    fn=greet,
    inputs=gr.Textbox(lines=2, placeholder="Please input your essay here to evaluate..."),
    outputs= gr.Highlightedtext(
      label = "Segmentation",
      combine_adjacent = True,
    )
    # .style(color_map={"Evaluate": "green"}),
)
demo.launch()