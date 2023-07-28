#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
import pickle
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../../'))

import pandas as pd

from allennlp.data.vocabulary import Vocabulary

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_path",
        default=(project_path / "pretrained_models/chinese-bert-wwm-ext").as_posix(),
        type=str
    )
    parser.add_argument('--hierarchical_labels_pkl', default='hierarchical_labels.pkl', type=str)
    parser.add_argument('--vocabulary', default='vocabulary', type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    with open(args.hierarchical_labels_pkl, 'rb') as f:
        hierarchical_labels = pickle.load(f)
    # print(hierarchical_labels)
    # 深度遍历
    token_to_index = OrderedDict()
    tasks = [hierarchical_labels]
    while len(tasks) != 0:
        task = tasks.pop(0)
        for parent, downstream in task.items():
            if isinstance(downstream, list):
                for label in downstream:
                    if pd.isna(label):
                        continue
                    label = '{}_{}'.format(parent, label)
                    token_to_index[label] = len(token_to_index)
            elif isinstance(downstream, OrderedDict):
                new_task = OrderedDict()
                for k, v in downstream.items():
                    new_task['{}_{}'.format(parent, k)] = v
                tasks.append(new_task)
            else:
                raise NotImplementedError

    vocabulary = Vocabulary(non_padded_namespaces=['tokens', 'labels'])
    for label, index in token_to_index.items():
        vocabulary.add_token_to_namespace(label, namespace='labels')

    vocabulary.set_from_file(
        filename=os.path.join(args.pretrained_model_path, 'vocab.txt'),
        is_padded=False,
        oov_token='[UNK]',
        namespace='tokens',
    )
    vocabulary.save_to_files(args.vocabulary)

    print('注意检查 Vocabulary 中标签的顺序与 hierarchical_labels 是否一致. ')
    return


if __name__ == '__main__':
    main()
