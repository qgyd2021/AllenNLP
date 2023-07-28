#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import copy
from itertools import chain
import os

from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary

from project_settings import project_path
from toolbox.allennlp_models.tagging.dataset_readers.bio_tagging_json_reader import BioTaggingJsonReader


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

    dataset_reader = BioTaggingJsonReader(
        token_indexers={
            "tokens": SingleIdTokenIndexer(
                namespace="tokens",
                lowercase_tokens=True,
                token_min_padding_length=0,
            )
        }
    )

    train_dataset = dataset_reader.read(args.train_subset)
    dev_dataset = dataset_reader.read(args.valid_subset)

    vocabulary = Vocabulary.from_instances(chain(train_dataset, dev_dataset))
    vocabulary.set_from_file(
        filename=os.path.join(args.pretrained_model_path, 'vocab.txt'),
        is_padded=False,
        oov_token='[UNK]',
        namespace='tokens',
    )

    label_to_index = vocabulary.get_token_to_index_vocabulary(namespace='labels')
    label_to_index = copy.deepcopy(label_to_index)
    # print(label_to_index)

    for label, idx in label_to_index.items():
        if label.startswith('B-'):
            inner_label = 'I-{}'.format(label[2:])
            if inner_label not in label_to_index.keys():
                # print(inner_label)
                vocabulary.add_token_to_namespace(token=inner_label, namespace='labels')

    vocabulary.save_to_files(args.vocabulary_dir)

    return


if __name__ == '__main__':
    main()
