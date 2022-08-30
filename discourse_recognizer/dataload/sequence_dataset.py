import numpy as np
from bisect import bisect_left
from discourse_recognizer.config import disc_type_to_ids, MIN_BEGIN_PROB, MAX_SEQ_LEN, cfg


class SeqDataset(object):
    def __init__(self, features, labels, groups, wordRanges, truePos):
        self.features = np.array(features, dtype=np.float32)
        self.labels = np.array(labels)
        self.groups = np.array(groups, dtype=np.int16)
        self.wordRanges = np.array(wordRanges, dtype=np.int16)
        self.truePos = np.array(truePos)


# Adapted from https://stackoverflow.com/questions/60467081/linear-interpolation-in-numpy-quantile
# This is used to prevent re-sorting to compute quantile for every sequence.
def sorted_quantile(array, q):
    array = np.array(array)
    n = len(array)
    index = (n - 1) * q
    left = np.floor(index).astype(int)
    fraction = index - left
    right = left
    right = right + (fraction > 0).astype(int)
    i, j = array[left], array[right]
    return i + (j - i) * fraction


def seq_dataset(disc_type, word_preds, pred_indices=None):
    begin_class_ids = [0, 1, 3, 5, 7, 9, 11, 13]  # O and B- labels index
    N_FEATURES = cfg["N_FEATURES"]

    window = pred_indices if pred_indices else range(len(word_preds))
    X = np.empty((int(1e6), N_FEATURES), dtype=np.float32)
    X_ind = 0
    y = []
    truePos = []
    wordRanges = []  # the start and end word index of the sequence in the text
    groups = []  # the integer index of the text where the sequence is found

    for text_i in window:
        text_preds = np.array(word_preds[text_i])
        num_words = len(text_preds)

        global_features, global_locs = [], []
        for dt in disc_type_to_ids:
            disc_begin, disc_inside = disc_type_to_ids[dt]

            gmean = (text_preds[:, disc_begin] + text_preds[:, disc_inside]).mean()
            global_features.append(gmean)
            global_locs.append(np.argmax(text_preds[:, disc_begin]) / float(num_words))

        disc_begin, disc_inside = disc_type_to_ids[disc_type]
        # The probability that a word corresponds to either a 'B'-egin or 'I'-nside token for a class
        prob_or = lambda text_word_preds: text_word_preds[:, disc_begin] + text_word_preds[:, disc_inside]

        # Iterate over every sub-sequence in the text
        quants = np.linspace(0, 1, 7)
        prob_begins = np.copy(text_preds[:, disc_begin])
        min_begin = MIN_BEGIN_PROB[disc_type]

        for pred_start in range(num_words):
            prob_begin = prob_begins[pred_start]
            if prob_begin > min_begin:
                begin_or_inside = []

                for pred_end in range(pred_start + 1, min(num_words + 1, pred_start + MAX_SEQ_LEN[disc_type] + 1)):
                    new_prob = prob_or(text_preds[pred_end - 1:pred_end])
                    insert_i = bisect_left(begin_or_inside, new_prob)
                    begin_or_inside.insert(insert_i, new_prob[0])

                    # Generate features for a word sub-sequence

                    # The length and position of start/end of the sequence
                    features = [pred_end - pred_start, pred_start / float(num_words), pred_end / float(num_words)]

                    # 7 evenly spaced quantiles of the distribution of relevant class probabilities for this sequence
                    features.extend(list(sorted_quantile(begin_or_inside, quants)))

                    # The proba that words on either edge of the current sub-sequence belong to the class of interest
                    features.append(prob_or(text_preds[pred_start - 1:pred_start])[0] if pred_start > 0 else 0)
                    features.append(prob_or(text_preds[pred_end:pred_end + 1])[0] if pred_end < num_words else 0)
                    features.append(prob_or(text_preds[pred_start - 2:pred_start - 1])[0] if pred_start > 1 else 0)
                    features.append(
                        prob_or(text_preds[pred_end + 1:pred_end + 2])[0] if pred_end < (num_words - 1) else 0)

                    # The probability that the first word corresponds to a 'B'-egin token
                    features.append(text_preds[pred_start, disc_begin])
                    features.append(text_preds[pred_start - 1, disc_begin])

                    if pred_end < num_words:
                        features.append(text_preds[pred_end, begin_class_ids].sum())
                    else:
                        features.append(1.0)

                    s = prob_or(text_preds[pred_start:pred_end])
                    features.append(np.argmax(s) / features[0])  # maximum point location on sequence
                    features.append(np.argmin(s) / features[0])  # minimum point location on sequence

                    instability = 0
                    if len(s) > 1:
                        instability = (np.diff(s) ** 2).mean()
                    features.append(instability)

                    features.extend(list(global_features))
                    features.extend(list([loc - features[1] for loc in global_locs]))

                    exact_match = None
                    true_pos = None

                    # For efficiency, use a numpy array instead of a list that doubles in size when full to conserve constant "append" time complexity
                    if X_ind >= X.shape[0]:
                        new_X = np.empty((X.shape[0] * 2, N_FEATURES), dtype=np.float32)
                        new_X[:X.shape[0]] = X
                        X = new_X
                    X[X_ind] = features
                    X_ind += 1

                    y.append(exact_match)
                    truePos.append(true_pos)
                    wordRanges.append((np.int16(pred_start), np.int16(pred_end)))
                    groups.append(np.int16(text_i))

    return SeqDataset(X[:X_ind], y, groups, wordRanges, truePos)
