#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
import sys
import time

from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.models.archival import archive_model, load_archive
from allennlp.predictors.text_classifier import TextClassifierPredictor
import torch
from allennlp.data.vocabulary import Vocabulary

from toolbox.allennlp.data.dataset_readers.text_classification_json import TextClassificationJsonReader
from project_settings import project_path
from toolbox.allennlp.data.tokenizers.sentence_piece_bpe_tokenizer import SentencePieceBPETokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        default="不好看，我觉得。",
        type=str
    )
    parser.add_argument(
        "--archive_file",
        default=(project_path / "trained_models/language_identification_bpe").as_posix(),
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

    json_dict = {
        "sentence": args.text
    }

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
