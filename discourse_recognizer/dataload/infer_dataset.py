import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def split_mapping(unsplit):
    # Return an array that maps character index to index of word in list of split() words
    splt = unsplit.split()
    offset_to_word_idx = np.full(len(unsplit), -1)
    txt_ptr = 0
    for split_index, full_word in enumerate(splt):
        while unsplit[txt_ptr:txt_ptr + len(full_word)] != full_word:
            txt_ptr += 1
        offset_to_word_idx[txt_ptr:txt_ptr + len(full_word)] = split_index
        txt_ptr += len(full_word)
    return offset_to_word_idx


class infer_dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, item):
        # Get text
        text = self.data.text[item]
        # Encode the text
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            padding=False,
            truncation=True,
            max_length=self.max_len
        )

        # Get the token ids maps Ex text = "Test unhappily"
        word_ids = encoding.word_ids()              # Ex [None, 0, 1, 1, 1, None]
        offsets = encoding['offset_mapping']  # Ex [(0, 0), (0, 4), (5, 8), (8, 11), (11, 14), (0, 0)]
        offset_to_word_idx = split_mapping(text)  # Ex [ 0, 0, 0, 0, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        split_word_ids = np.full(len(word_ids), -1) # Ex [-1, -1, -1, -1, -1, -1]

        # Iterate in reverse to label whitespace tokens until a Begin token is encountered
        for token_idx, word_idx in reversed(list(enumerate(word_ids))):
            if word_idx is not None:
                if offsets[token_idx][0] != offsets[token_idx][1]:
                    # Choose the split word that shares the most characters with the token if any
                    split_idxs = offset_to_word_idx[offsets[token_idx][0]:offsets[token_idx][1]]
                    split_index = stats.mode(split_idxs[split_idxs != -1]).mode[0] if len(
                        np.unique(split_idxs)) > 1 else split_idxs[0]

                    if split_index != -1:
                        split_word_ids[token_idx] = split_index

        # Convert to torch tensor
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item["wids"] = torch.as_tensor(split_word_ids)  # after code block split_word_idx = [-1, 0, 1, 1, 1, -1]
        return item

    def __len__(self):
        return self.len


class CustomCollate:
    def __init__(self, tokenizer, sliding_window=None):
        self.tokenizer = tokenizer
        self.sliding_window = sliding_window

    def __call__(self, data):
        """
        need to collate: input_ids, attention_mask
        input_ids is padded with 1, attention_mask 0
        """
        bs = len(data)
        lengths = []
        for i in range(bs):
            lengths.append(len(data[i]['input_ids']))
        max_len = max(lengths)

        if self.sliding_window is not None and max_len > self.sliding_window:
            max_len = int((np.floor(max_len / self.sliding_window - 1e-6) + 1) * self.sliding_window)

        # Always pad the right side
        input_ids, attention_mask, labels, BIO_labels, discourse_labels = [], [], [], [], []

        wids = []
        for i in range(bs):
            input_ids.append(F.pad(data[i]['input_ids'], (0, max_len - lengths[i]), value=self.tokenizer.pad_token_id))
            attention_mask.append(F.pad(data[i]['attention_mask'], (0, max_len - lengths[i]), value=0))
            wids.append(F.pad(data[i]['wids'], (0, max_len - lengths[i]), value=-1))

        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        wids = torch.stack(wids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "wids": wids
        }
