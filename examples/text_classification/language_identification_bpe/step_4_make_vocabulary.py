#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import copy
from itertools import chain
import os
from pathlib import Path

from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary

from toolbox.allennlp.data.dataset_readers.text_classification_json import TextClassificationJsonReader
from toolbox.allennlp.data.tokenizers.sentence_piece_bpe_tokenizer import SentencePieceBPETokenizer
from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bpe_tokenizer_file",
        default="data_dir/bpe_tokenizer.model",
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
        tokenizer=SentencePieceBPETokenizer(
            bpe_tokenizer_file=args.bpe_tokenizer_file
        ),
    )

    train_dataset = dataset_reader.read(args.train_subset)
    valid_dataset = dataset_reader.read(args.valid_subset)

    vocabulary = Vocabulary.from_instances(chain(train_dataset, valid_dataset))
    vocabulary.save_to_files(args.vocabulary_dir)

    return


if __name__ == '__main__':
    main()
