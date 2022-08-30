import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from cuml import ForestInference
from discourse_recognizer.config import cfg, features_dict, discourses
from discourse_recognizer.dataload.sequence_dataset import seq_dataset


def get_tree_models(N_XGB_FOLDS=1):
    xgb_models, lgb_models = dict(), dict()

    for d in discourses:
        model_list = []
        for f in range(N_XGB_FOLDS):
            xgb_model = ForestInference.load(os.path.join(cfg["tree_models_folder"], f"xgb_{d}_{f}.json"), output_class=True,
                                             model_type="xgboost_json")
            model_list.append(xgb_model)
        xgb_models[d] = model_list

        model_list = []
        for f in range(N_XGB_FOLDS):
            lgb_model = ForestInference.load(os.path.join(cfg["tree_models_folder"], f"lgb_{d}_{f}.txt"), output_class=True,
                                             model_type="lightgbm")
            model_list.append(lgb_model)
        lgb_models[d] = model_list

    return xgb_models, lgb_models


def get_tp_prob(testDs, disc_type, xgb_models, lgb_models):

    if testDs.features.shape[0] == 0:
        return np.array([])

    pred = np.mean([clf.predict_proba(testDs.features[:, features_dict[disc_type]].astype("float32"))[:, 1] for clf in xgb_models[disc_type]], axis=0)/2
    pred += np.mean([clf.predict_proba(testDs.features[:, features_dict[disc_type]].astype("float32"))[:, 1] for clf in lgb_models[disc_type]], axis=0)/2

    return pred


def get_test_dataframe(texts:list):
    # Convert texts to dataframe
    test_id, test_texts = [], []
    for i, essay_text in enumerate(texts):
        test_id.append(i)
        test_texts.append(essay_text)
    test_df = pd.DataFrame({"id": test_id, "text": test_texts})
    return test_df


def predict_strings(disc_type, probThresh, test_groups, test_texts, word_preds, xgb_models, lgb_models):
    string_preds = []

    # Average the probability predictions of a set of classifiers

    predict_df = test_texts
    text_df = test_texts

    for text_idx in tqdm(test_groups):
        # The probability of true positive and (start,end) of each sub-sequence in the current text

        testDs = seq_dataset(disc_type, word_preds, [text_idx])

        prob_tp_curr = get_tp_prob(testDs, disc_type, xgb_models, lgb_models)
        word_ranges_curr = testDs.wordRanges[testDs.groups == text_idx]

        split_text = text_df.loc[text_df.id == predict_df.id.values[text_idx]].iloc[0].text.split()
        full_preds = np.zeros(len(split_text))
        # Include the sub-sequence predictions in order of predicted probability
        for prob, wordRange in reversed(sorted(zip(prob_tp_curr, [tuple(wr) for wr in word_ranges_curr]))):

            # Until the predicted probability is lower than the tuned threshold
            if prob < probThresh: break

            intersect = np.sum(full_preds[wordRange[0]:wordRange[1]])
            total = wordRange[1] - wordRange[0]
            condition = intersect / total <= 0.15

            if condition:
                full_preds[wordRange[0]:wordRange[1]] = 1
                string_preds.append((predict_df.id.values[text_idx], disc_type,
                                     ' '.join(map(str, list(range(wordRange[0], wordRange[1]))))))
    return string_preds


def sub_df(string_preds):
    return pd.DataFrame(string_preds, columns=['id','class','predictionstring'])







