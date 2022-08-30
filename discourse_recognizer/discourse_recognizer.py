import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from .models.transformer_model import TransformerModel, SlidingWindowTransformerModel
from .dataload.infer_dataset import infer_dataset, CustomCollate
from .tools.predict import get_test_dataframe, get_tree_models, predict_strings, sub_df
from .tools.inference import inference
from .config import thresholds, disc_type_to_ids, cfg


class DiscourseRecognizer:
    def __init__(self, config=cfg) -> None:
        self.config = config
        self.xgb_models, self.lgb_models = get_tree_models(self.config["N_XGB_FOLDS"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["DOWNLOADED_MODEL_PATH"])
        self.model = SlidingWindowTransformerModel(self.config["DOWNLOADED_MODEL_PATH"], rnn="GRU").to(
            self.config["device"])

    def process(self, essay: str):
        """_summary_
        Args:
            essay (str): _description_
        Returns:
            List[dict]: _description_

        Returns a list of dict, each dict represents a discourse text
        [
            {
                "start": int, -> the start position (character index)
                "end": int, -> the end position (character index)
                "type": str, -> discourse type
            },
            ...
        ]
        """
        test_texts = [essay]
        test_df = get_test_dataframe(test_texts)
        # print("TEST DF: ", test_df)
        test_dataset = infer_dataset(test_df, self.tokenizer, max_len=self.config["max_length"])
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.config["valid_batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=CustomCollate(self.tokenizer, 512)
        )
        test_words_preds = inference(test_dataloader, [0], self.model, path=self.config["LOAD_MODEL_FROM"])
        groups = range(len(test_words_preds))

        output_df = pd.concat([
            sub_df(
                predict_strings(
                    disc_type,
                    thresholds[disc_type],
                    groups,
                    test_df,
                    test_words_preds,
                    self.xgb_models,
                    self.lgb_models
                )
            ) for disc_type in disc_type_to_ids
        ])

        output_list = []
        for i in range(len(output_df)):
            discourse_dict = {
                "start": int(str(output_df.iloc[i]["predictionstring"]).split()[0]),
                "end": int(str(output_df.iloc[i]["predictionstring"]).split()[-1]),
                "type": str(output_df.iloc[i]["class"])
            }
            output_list.append(discourse_dict)

        return output_list


if __name__ == "__main__":
    import time

    disc_re = DiscourseRecognizer(config=cfg)

    while (1):
        essay = str(input("Essay:"))
        start = time.time()
        output_list = disc_re.process(essay)
        for item in output_list:
            print(item)
        end = time.time()
        print("Process run time: {} s".format(start - end))

        stop = str(input("End? (Y,N): "))
        if stop.lower() == "y":
            break




