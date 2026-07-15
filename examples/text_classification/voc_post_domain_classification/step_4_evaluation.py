#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, "../../../"))

from allennlp.models.archival import load_archive
from allennlp.predictors.text_classifier import TextClassifierPredictor

import pandas as pd
from tqdm import tqdm

from toolbox.allennlp.archive_overrides import default_tokenizer_overrides
from toolbox.allennlp.data.dataset_readers.text_classification_json import TextClassificationJsonReader
from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive_file",
        default=(project_path / "trained_models/voc_post_domain_textcnn").as_posix(),
        type=str,
    )
    parser.add_argument(
        "--train_subset",
        default="data_dir/train.jsonl",
        type=str,
    )
    parser.add_argument(
        "--valid_subset",
        default="data_dir/valid.jsonl",
        type=str,
    )
    parser.add_argument(
        "--evaluation_output_file",
        default="data_dir/evaluation.xlsx",
        type=str,
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    archive = load_archive(
        archive_file=args.archive_file,
        overrides=default_tokenizer_overrides(args.archive_file),
    )

    predictor = TextClassifierPredictor(
        model=archive.model,
        dataset_reader=archive.dataset_reader,
    )

    result = list()

    for subset, filename in [("train", args.train_subset), ("valid", args.valid_subset)]:
        total_count = 0
        correct_count = 0
        with tqdm() as progress_bar:
            with open(filename, "r", encoding="utf-8") as f:
                for row in f:
                    row = json.loads(row)
                    text = row["text"]
                    ground_true = row["label"]

                    json_dict = {
                        "sentence": text,
                    }

                    outputs = predictor.predict_json(json_dict)
                    predict = outputs["label"]
                    probs = outputs["probs"]
                    prob = round(max(probs), 4)
                    correct = 1 if ground_true == predict else 0

                    result.append({
                        "subset": subset,
                        "text": text,
                        "ground_true": ground_true,
                        "predict": predict,
                        "prob": prob,
                        "correct": correct,
                    })
                    total_count += 1
                    correct_count += correct
                    accuracy = round(correct_count / total_count, 4)

                    progress_bar.update(1)
                    progress_bar.set_postfix({"accuracy": accuracy, "subset": subset})

    result = pd.DataFrame(result)
    result.to_excel(
        args.evaluation_output_file,
        index=False,
        engine="xlsxwriter",
    )
    return


if __name__ == "__main__":
    main()
