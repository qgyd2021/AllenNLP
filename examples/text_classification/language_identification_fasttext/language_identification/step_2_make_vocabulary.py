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
        "--pretrained_model_path",
        default=(project_path / "pretrained_models/chinese-bert-wwm-ext").as_posix(),
        type=str
    )
    parser.add_argument(
        "--train_subset",
        default="data_dir/train.jsonl",
        type=str
    )
    parser.add_argument(
        "--valid_subset",
        default="data_dir/valid.jsonl",
        type=str
    )
    parser.add_argument(
        "--vocabulary_dir",
        default="vocabulary",
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    vocabulary_dir = Path(args.vocabulary_dir)
    vocabulary_dir.parent.mkdir(parents=True, exist_ok=True)

    dataset_reader = TextClassificationJsonReader(
        token_indexers={
            "tokens": SingleIdTokenIndexer(
                namespace="tokens",
                lowercase_tokens=True,
                token_min_padding_length=5,
            ),
        },
        tokenizer=PretrainedTransformerTokenizer(
            model_name=args.pretrained_model_path,
        ),
        max_sequence_length=256,
    )

    train_dataset = dataset_reader.read(args.train_subset)
    valid_dataset = dataset_reader.read(args.valid_subset)

    vocabulary = Vocabulary.from_instances(chain(train_dataset, valid_dataset))
    vocabulary.set_from_file(
        filename=os.path.join(args.pretrained_model_path, "vocab.txt"),
        is_padded=False,
        oov_token="[UNK]",
        namespace="tokens",
    )
    vocabulary.save_to_files(args.vocabulary_dir)

    return


if __name__ == '__main__':
    main()
