#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import copy
from itertools import chain
import os

from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary

from project_settings import project_path
from toolbox.allennlp_models.pair_classification.dataset_readers.pair_classification_json import PairClassificationJsonReader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_path",
        default=(project_path / "pretrained_models/chinese-bert-wwm-ext").as_posix(),
        type=str
    )
    parser.add_argument(
        "--train_subset",
        default="data_dir/train.json",
        type=str
    )
    parser.add_argument(
        "--valid_subset",
        default="data_dir/valid.json",
        type=str
    )
    parser.add_argument(
        "--vocabulary_dir",
        default="data_dir/vocabulary",
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    dataset_reader = PairClassificationJsonReader(
        tokenizer=PretrainedTransformerTokenizer(
            model_name=args.pretrained_model_path,
        ),
        token_indexers={
            "premise": SingleIdTokenIndexer(
                namespace="tokens",
                lowercase_tokens=True,
                token_min_padding_length=5,
            ),
            "hypothesis": SingleIdTokenIndexer(
                namespace="tokens",
                lowercase_tokens=True,
                token_min_padding_length=5,
            )
        }
    )

    train_dataset = dataset_reader.read(args.train_subset)
    dev_dataset = dataset_reader.read(args.valid_subset)

    vocabulary = Vocabulary.from_instances(chain(train_dataset, dev_dataset))
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
