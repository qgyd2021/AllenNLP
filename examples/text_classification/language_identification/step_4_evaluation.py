#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
from pathlib import Path
import sys
import time

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, "../../../"))

from allennlp.models.archival import archive_model, load_archive
from allennlp.predictors.text_classifier import TextClassifierPredictor
from allennlp.training.util import evaluate

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from toolbox.allennlp.data.dataset_readers.text_classification_json import TextClassificationJsonReader
from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive_file",
        default=(project_path / "trained_models/language_identification").as_posix(),
        type=str
    )
    parser.add_argument(
        "--train_subset",
        default="train.jsonl",
        type=str
    )
    parser.add_argument(
        "--valid_subset",
        default="valid.jsonl",
        type=str
    )
    parser.add_argument(
        "--evaluation_output_file",
        default="evaluation.xlsx",
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    archive = load_archive(archive_file=args.archive_file)

    predictor = TextClassifierPredictor(
        model=archive.model,
        dataset_reader=archive.dataset_reader,
    )

    result = list()

    total_count = 0
    correct_count = 0
    with tqdm() as progress_bar:
        with open(args.valid_subset, "r", encoding="utf-8") as f:
            subset = Path(args.valid_subset).stem
            for row in f:
                row = json.loads(row)
                text = row["text"]
                ground_true = row["label"]
                split = row["split"]

                json_dict = {
                    "sentence": text
                }

                outputs = predictor.predict_json(
                    json_dict
                )
                predict = outputs["label"]
                probs = outputs["probs"]
                prob = round(max(probs), 4)
                correct = 1 if ground_true == predict else 0

                result.append({
                    "subset": subset,
                    "split": split,
                    "text": text,
                    "ground_true": ground_true,
                    "predict": predict,
                    "prob": prob,
                    "correct": correct
                })
                total_count += 1
                correct_count += correct
                accuracy = correct_count / total_count
                accuracy = round(accuracy, 4)

                progress_bar.update(1)
                progress_bar.set_postfix({"accuracy": accuracy, "subset": subset})

    result = pd.DataFrame(result)
    result.to_excel(args.evaluation_output_file, index=False, encoding="utf_8_sig", engine="xlsxwriter")

    return


if __name__ == '__main__':
    main()
