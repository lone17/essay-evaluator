import numpy as np
from discourse_recognizer.config import disc_type_to_ids


def remove_overlap(output_list, words_preds):
    # Post-process remove overlaps
    output_list = sorted(output_list, key=lambda item: item["start"])
    test_words_preds = np.array(words_preds)

    for i in range(len(output_list) - 1):
        current_end = output_list[i]["end"]
        current_class = output_list[i]["type"]
        curr_begin, curr_inside = disc_type_to_ids[current_class]

        next_start = output_list[i + 1]["start"]
        next_class = output_list[i + 1]["type"]
        next_begin, next_inside = disc_type_to_ids[next_class]

        if current_end >= next_start:

            prob_curr_class = test_words_preds[0][current_end: next_start + 1, curr_begin] + test_words_preds[0][
                                                                                             current_end: next_start + 1,
                                                                                             curr_inside]
            prob_next_class = test_words_preds[0][current_end: next_start + 1, next_begin] + test_words_preds[0][
                                                                                             current_end: next_start + 1,
                                                                                             next_inside]

            if prob_curr_class.sum() > prob_next_class.sum():
                output_list[i]["end"] = next_start
                output_list[i + 1]["start"] += 1
            else:
                output_list[i + 1]["start"] = current_end
                output_list[i]["end"] -= 1
            # output_list[i+1]["start"] = current_end
            # output_list[i]["end"] -= 1

    index = 0
    while index < len(output_list):
        if output_list[index]["start"] >= output_list[index]["end"]:
            output_list.pop(index)
        else:
            index += 1

    return output_list
