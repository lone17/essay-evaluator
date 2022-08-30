import gc
import numpy as np
from discourse_recognizer.config import cfg, labels_to_ids
from tqdm import tqdm
import torch


# Returns per-word, mean class prediction probability over all tokens corresponding to each word
def inference(data_loader, model_ids, model, path):
    gc.collect()
    torch.cuda.empty_cache()

    ensemble_preds = np.zeros((len(data_loader.dataset), cfg['max_length'], len(labels_to_ids)), dtype=np.float32)
    wids = np.full((len(data_loader.dataset), cfg['max_length']), -100)
    for model_i, model_id in enumerate(model_ids):
        model.load_state_dict(torch.load(f'{path}/fold{model_id}.pt', map_location=cfg['device']))

        # put model in training mode
        model.eval()
        for batch_i, batch in tqdm(enumerate(data_loader)):

            if model_i == 0:
                wids[batch_i * cfg['valid_batch_size']:(batch_i + 1) * cfg['valid_batch_size'],
                :batch['wids'].shape[1]] = batch['wids'].numpy()

            # MOVE BATCH TO GPU AND INFER
            ids = batch["input_ids"].to(cfg['device'])
            mask = batch["attention_mask"].to(cfg['device'])
            with torch.no_grad():
                outputs = model(ids, attention_mask=mask)
            all_preds = torch.nn.functional.softmax(outputs[0], dim=2).cpu().detach().numpy()
            ensemble_preds[batch_i * cfg['valid_batch_size']:(batch_i + 1) * cfg['valid_batch_size'],
            :all_preds.shape[1]] += all_preds

            del ids
            del mask
            del outputs
            del all_preds

        gc.collect()
        torch.cuda.empty_cache()

    ensemble_preds /= len(model_ids)
    predictions = []  # list of prediction for samples (num text, num words in text, len(labels))
    # INTERATE THROUGH EACH TEXT AND GET PRED
    for text_i in range(ensemble_preds.shape[0]):
        token_preds = ensemble_preds[text_i]

        prediction = []  # List of prediction for words in sequence with the shape of (num words in seq, len(labels))
        previous_word_idx = -1
        prob_buffer = []
        word_ids = wids[text_i][wids[text_i] != -100]  # Ex [-1, 0, 1, 1, 1, -1]
        for idx, word_idx in enumerate(word_ids):
            if word_idx == -1:
                pass
            elif word_idx != previous_word_idx:
                if prob_buffer:
                    prediction.append(np.mean(prob_buffer, dtype=np.float32, axis=0))
                    prob_buffer = []
                prob_buffer.append(token_preds[idx])
                previous_word_idx = word_idx
            else:
                prob_buffer.append(token_preds[idx])
        prediction.append(np.mean(prob_buffer, dtype=np.float32, axis=0))

        predictions.append(prediction)

    gc.collect()
    torch.cuda.empty_cache()
    return predictions
