#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import sys
import time

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, "../../../"))

from allennlp.models.archival import load_archive
from allennlp.predictors.text_classifier import TextClassifierPredictor

from project_settings import project_path
from toolbox.allennlp.archive_overrides import default_tokenizer_overrides
from toolbox.allennlp.data.dataset_readers.text_classification_json import TextClassificationJsonReader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        # default="这是一段用于测试预测效果的帖子内容。",
        default="关于新款雷柏V3 PRO V2的一些使用感受",
        type=str,
    )
    parser.add_argument(
        "--archive_file",
        default=(project_path / "trained_models/voc_post_domain_textcnn").as_posix(),
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

    json_dict = {
        "sentence": args.text,
    }

    begin_time = time.time()
    outputs = predictor.predict_json(json_dict)
    label = outputs["label"]
    probs = outputs["probs"]

    max_prob = round(max(probs), 4)
    print("label: {}".format(label))
    print("prob: {}".format(max_prob))

    print("time cost: {}".format(time.time() - begin_time))
    return


if __name__ == "__main__":
    main()
