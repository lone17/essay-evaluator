from essay_evaluator import EssayEvaluator
import gradio as gr

evaluator = EssayEvaluator(
    discourse_recognizer_params={},
    discourse_evaluator_params={"weights": "./discourse_evaluator/pretrained/bert-base.pth"}
)

def splitter(start, end, str):
    pieces = str.split()
    return " ".join(pieces[start : end + 1])

def greet(essay):
    essay_list = evaluator.process(essay)
    essay_list.append({"start" : -1, "end": 0, "type": None, "effectiveness": None, "score": None})
    sorted_essay = sorted(essay_list, key=lambda d: d['start'])
    outputText = []
    for i in range(1, len(sorted_essay)):
        if (sorted_essay[i]['start'] > sorted_essay[i - 1]['end'] + 1):
            segmentText = splitter(sorted_essay[i - 1]['end'] + 1, sorted_essay[i]['start'] - 1, essay) + '\n'
            outputText.append((segmentText, None))
        segmentText = f"[{sorted_essay[i]['effectiveness']} | {sorted_essay[i]['score']:.2f}]\n"
        outputText.append((segmentText, sorted_essay[i]['type']))
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
      combine_adjacent = False,
      show_legend = True
    ),
    examples=[
        [
"""Dear Senator

I feel that electoral college should stay becuase it set's straight the manege for the peoples votes for the president. Even thow if a tie the election would be thrown to the house of represenatives were they decide on the new president. The electoral college is a process its not a place. It shows that the founding fathers established it for the compromise between elections between the president by a vote in congress. The electoral college consists of 538 electors. Plus a majority of 270 electoral votes is required to elect the president .It say's in the passage that under the 23rd Amendment of the constiton that the district of colombia is allocated 3 electors and treated like a state for the purposes of electoral college. On many cases 60 percent of voters would perfer a direct election. The electoral college system states that voters vote not for the president,but for a state of eletors who elect the president. I think that they should have the right to elect the president that would make this nation grow stonger not to make mistakes and lead us to problems which they can decide to choose the future of the united states. Many people have diffrent opinions as me but not all will get the same results as they wanted. The effects of the electoral college has been giving us positive values. As we live on since back of the elections day we are still here becuase of thow,s elections it made us who we are american citizions of the USA.History depends on the future rights of the president thats why the electoral college is there to help make that happen."""
        ],

        [
"""Imagine for a minute that you are at school, on three hours of sleep with an innate ability to care for anything at this present time. You wish that you could just crawl back into your hovel and sleep for a few more hours. Despite your most earnest wishes, however, your required presence at school forbids such activities from transpiring. Luckily for you, there is yet a ray of hope. Some schools have begun to offer distance learning as a way for students to attend from home, usually via online means. I believe that students stand to gain from this alternate method of attendance. I think this because distance learning decreases stress on students, reduces bullying, and is a healthy alternative for students with social anxiety. Furthermore, distance learning provides the student with an enhanced ability to pursue self-education.

For many students, school can be the progenitor of a stressful and mentally debilitating environment. This, consequentially, leads to a decrease in a student's mental and even physical health. This negative environment is usually achieved though an overabundance of homework and classwork, as well as assorted busy-work and tight deadlines. Through that invention which is distance learning, students will have an option to choose a slower-paced working environment. When working from a computer, classwork and homework lose distinction, and just become "work," as there is no assigned place to start and finish it. This alone would do wonders for the mental health of students. To use an example from my own experiences with distance learning, I have found assignments done online to be more enjoyable and less stressful than those done in a classroom.

Another factor that would considerably benefit student health is the drastic reduction in instances of bullying that distance learning would bring. For the uninitiated, bullying is when one person engages in harmful acts repeatedly and over an extended period of time against another person. Schools have only served to increase instances of bullying through zero-tolerance policies, in which the victim is punished equally as much as the aggressor. This also leads to students being afraid to report bullying, as they would also risk punishment. Distance learning would reduce bullying by putting "distance" between a bully and a victim of bullying. While cyberbulling would still be present, as well as an issue, it is considerably less damaging when compared to the verbal and physical confrontations that can be seen on school grounds.

Distance learning would also be considered a pious alternative of sorts for those students who do not hold social interaction in high esteem, myself included. While most students can be considered to be outgoing in one way or another, there is a large minority of students that range from being just generally introverted to having crippling social anxiety. Their needs are usually overlooked in most schools, where the social climate in many cases is best described as "riotous." The prospect of attending classes from home for these students would be incredibly enticing, and would be calming for them. To use another personal example, I often find myself to be more productive at home than I do at school, in part because of the disruptive social climate that classrooms have.

Perhaps the most consequential result of distance learning would be an enhancement in the student's ability to pursue self-education. Self-education is, unsurprisingly, when a student independently seeks out and digests information, whether it is needed for school or not. Self-education can also have a profound effect on students, whether by giving them a higher grade on a test, or leading them to decide on a college major based off of a topic that they find to be interesting. To use myself an example, I have been educating myself on matters of Astronomy, Military History, and Geography for approximately 7 years now. In that time, I have participated in three state-level geography competitions, the second of which ranked me as third in the state in knowledge of Ancient Geography. In Astronomy, my self-education allowed me to get full marks on almost every Astronomy-related assignment in school, and in Military History, perhaps my strongest subject, I have connected myself with various historians and am currently making inroads into publishing work of my own design in a Military Journal. With distance learning, myself and every other like-minded student would have more time to pursue these subjects.

In conclusion, it is my belief that students would benefit immensely from distance learning. Working from home would take students out of a stressful school environment, and by extension protect them from bullying. That same home environment would also be better suited for more introverted students, who might feel anxious while being in school. Distance learning would also allow students to better pursue self-education, which would have positive effects for them presently and also later in life. Overall, distance learning stands to enrich the lives of students, while also serving to better their mental and physical health."""
        ]
    ],
    description=(
                "<div>"
                "<img  src='file/banner_demoday2.jpg' alt='image One'>"
                + "</div>"
                ),
)
demo.launch(share=True)