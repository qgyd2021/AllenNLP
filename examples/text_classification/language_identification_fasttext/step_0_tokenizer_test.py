#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import copy
from itertools import chain
import os
from pathlib import Path
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, "../../../"))

from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary

from toolbox.allennlp.data.dataset_readers.text_classification_json import TextClassificationJsonReader
from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        default="叔本华信仰什么宗教？",
        type=str
    )
    parser.add_argument(
        "--pretrained_model_path",
        default=(project_path / "pretrained_models/huggingface/bert-base-multilingual-cased").as_posix(),
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    tokenizer = PretrainedTransformerTokenizer(
        model_name=args.pretrained_model_path,
    )

    tokens = tokenizer.tokenize(args.text)
    print(tokens)

    return


if __name__ == '__main__':
    main()
