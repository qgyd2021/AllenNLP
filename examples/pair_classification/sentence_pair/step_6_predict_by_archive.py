#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://huggingface.co/allenai/bidaf-elmo
https://huggingface.co/docs/hub/allennlp

"""
import argparse
import time

from allennlp.models.archival import archive_model, load_archive
from allennlp.predictors.predictor import Predictor
from allennlp_models.pair_classification.models.decomposable_attention import DecomposableAttention
from allennlp_models.pair_classification.predictors.textual_entailment import TextualEntailmentPredictor

from project_settings import project_path
from toolbox.allennlp_models.pair_classification.dataset_readers.pair_classification_json import PairClassificationJsonReader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--premise",
        default="一般电话确认要等多久。",
        type=str
    )
    parser.add_argument(
        "--hypothesis",
        default="一般多久才会打电话来",
        type=str
    )
    parser.add_argument(
        "--archive_file",
        default=(project_path / "trained_models/ccks2018_task3_pair_classification").as_posix(),
        type=str
    )
    parser.add_argument(
        "--pretrained_model_path",
        default=(project_path / "pretrained_models/chinese-bert-wwm-ext").as_posix(),
        type=str
    )
    parser.add_argument(
        "--predictor_name",
        default="textual_entailment",
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    archive = load_archive(archive_file=args.archive_file)

    predictor = Predictor.from_archive(archive, predictor_name=args.predictor_name)

    json_dict = {"premise": args.premise, "hypothesis": args.hypothesis}

    begin_time = time.time()
    outputs = predictor.predict_json(
        json_dict
    )
    label = outputs["label"]
    print(label)

    print('time cost: {}'.format(time.time() - begin_time))
    return


if __name__ == '__main__':
    main()
