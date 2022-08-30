import os
import torch
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

discourses = ['Lead', 'Position', 'Evidence', 'Claim', 'Concluding Statement', 'Counterclaim', 'Rebuttal']

# CREATE DICTIONARIES THAT WE CAN USE DURING TRAIN AND INFER
output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim',
                 'I-Counterclaim',
                 'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement',
                 'I-Concluding Statement']

labels_to_ids = {v: k for k, v in enumerate(output_labels)}
ids_to_labels = {k: v for k, v in enumerate(output_labels)}

disc_type_to_ids = {'Evidence': (11, 12), 'Claim': (5, 6), 'Lead': (1, 2), 'Position': (3, 4), 'Counterclaim': (7, 8),
                    'Rebuttal': (9, 10), 'Concluding Statement': (13, 14)}

ensemble_weights = {"Rebuttal": 0.65,
                    "Counterclaim": 0.75,
                    "Concluding Statement": 0.60,
                    "Claim": 0.65,
                    "Evidence": 0.60,
                    "Position": 0.75,
                    "Lead": 0.70
                    }

features_dict = {
    'Lead': [i for i in range(34)],
    'Position': [i for i in range(34)],
    'Evidence': [i for i in range(20)],
    'Claim': [i for i in range(20)],
    'Concluding Statement': [i for i in range(34)],
    'Counterclaim': [i for i in range(17)] + [i for i in range(27, 34)],
    'Rebuttal': [i for i in range(17)]
}

thresholds = {
    'Lead': 0.66,
    'Position': 0.56,
    'Evidence': 0.57,
    'Claim': 0.54,
    'Concluding Statement': 0.56,
    'Counterclaim': 0.7,
    'Rebuttal': 0.74}


# The minimum probability prediction for a 'B'egin class for which we will evaluate a word sequence
MIN_BEGIN_PROB = {
    'Claim': .35 * 0.8,
    'Concluding Statement': .15 * 1.0,
    'Counterclaim': .04 * 1.25,
    'Evidence': .1 * 0.8,
    'Lead': .32 * 1.0,
    'Position': .25 * 0.8,
    'Rebuttal': .01 * 1.25,
}

# Use 99.5% of the distribution of lengths for a discourse type as maximum.
# Increasing this constraint makes this step slower but generally increases performance.
# train_df = pd.read_csv("./input/feedback-prize-2021/train.csv")
# MAX_SEQ_LEN = {}
# train_df['len'] = train_df['predictionstring'].apply(lambda x: len(x.split()))
# max_lens = train_df.groupby('discourse_type')['len'].quantile(.995)
# for disc_type in disc_type_to_ids:
#     MAX_SEQ_LEN[disc_type] = int(max_lens[disc_type])

MAX_SEQ_LEN = {
    'Evidence': 299,
    'Claim': 55,
    'Lead': 197,
    'Position': 59,
    'Counterclaim': 95,
    'Rebuttal': 116,
    'Concluding Statement': 191
}

cfg = {
    'LOAD_MODEL_FROM': './discourse_recognizer/input/fp-test78',
    'DOWNLOADED_MODEL_PATH': './discourse_recognizer/input/deberta-xlarge',
    'tree_models_folder': './discourse_recognizer/input/student-writing-7322/',


    'N_FEATURES': 34,
    'N_XGB_FOLDS': 1,

    'model_name': '',
    'max_length': 2048,
    'train_batch_size': 4,
    'valid_batch_size': 4,
    'epochs': 5,
    'learning_rates': [2.5e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-7],
    'max_grad_norm': 10,
    'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
}

if __name__ == "__main__":
    for k, v in cfg.items():
        print(f"{k} : {v}")
    print("=" * 50)
    print(MAX_SEQ_LEN)