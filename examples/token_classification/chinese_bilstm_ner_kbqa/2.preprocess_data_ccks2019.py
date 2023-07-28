#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
from pathlib import Path
import re
from typing import List, Union

from tqdm import tqdm

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default=(project_path / "examples/token_classification/chinese_bilstm_ner_kbqa/data_dir/CKBQA/data/ccks2019").as_posix(),
        type=str
    )
    parser.add_argument(
        "--pre_train_subset",
        default="data_dir/kbqa-train.json",
        type=str
    )
    parser.add_argument(
        "--pre_valid_subset",
        default="data_dir/kbqa-valid.json",
        type=str
    )
    parser.add_argument(
        "--pre_test_subset",
        default="data_dir/kbqa-test.json",
        type=str
    )
    args = parser.parse_args()
    return args


def max_length_common_substring(string1: str, string2: str) -> Union[None, dict]:
    common_substring_list = []

    for i in range(1, len(string1) + 1):
        for j in range(len(string1) + 1 - i):
            substring = string1[j:j + i]

            if substring in string2:
                common_substring_list.append({
                    'substring': substring,
                    'begin': j,
                    'end': j + i
                })

    if len(common_substring_list) == 0:
        return None

    result = common_substring_list[-1]
    return result


def process_point(block_list: List[str]):
    result = list()
    for b in block_list:
        if len(b) == 0:
            continue

        match = re.match(r'(\?\w)\.(\?\w)', b, flags=re.IGNORECASE)
        if match:
            result.append(match.group(1))
            result.append('.')
            result.append(match.group(2))
        elif b.endswith('.'):
            result.append(b[:-1])
            result.append('.')
        elif b.startswith('{') and len(b) != 1:
            result.append('{')
            result.append(b[1:])
        else:
            result.append(b)
    return result


def process_mention_for_one_hop(question: str, block_list: List[str]):
    subject = block_list[1]
    relation = block_list[2]
    object_ = block_list[3]

    mention_labels = list()
    if not subject.startswith('?'):
        mention = max_length_common_substring(question, subject[1:-1])
        # mention can be None
        if mention is not None:
            mention_labels.append(mention)

    if not object_.startswith('?'):
        mention = max_length_common_substring(question, object_[1:-1])
        # mention can be None
        if mention is not None:
            mention_labels.append(mention)

    return relation, mention_labels


def main():
    args = get_args()

    output_file_map = {
        "train": args.pre_train_subset,
        "valid": args.pre_valid_subset,
        "test": args.pre_test_subset,
    }

    data_path = Path(args.data_path)

    for filename in tqdm(data_path.glob("*.txt")):

        _, fn = os.path.split(filename)
        basename, ext = os.path.splitext(fn)

        output_file = output_file_map[basename]

        json_f = open(output_file, 'a+', encoding='utf-8')
        with open(filename, 'r', encoding='utf-8') as f:
            row = dict()
            question: str = None

            for line in f:
                line = str(line).strip()
                if len(line) != 0 and ord(line[0]) == 65279:
                    line = line[1:]

                if line.startswith('q'):
                    if len(row) != 0:
                        row = json.dumps(row, ensure_ascii=False)
                        json_f.write('{}\n'.format(row))
                        row = dict()
                        question = None

                    row['source'] = 'ccks2019'
                    row['origin_question'] = line
                    question_index, question = line.split(':', maxsplit=1)
                    row['question_index'] = question_index
                    row['question'] = question

                if line.startswith('select'):
                    row['SPARQL'] = line

                    _, where = line.split('where', maxsplit=1)

                    block_list = where.split(' ')
                    block_list = process_point(block_list)

                    if len(block_list) == 6:
                        row['n_hop'] = '单跳问句'
                        relation, mention_labels = process_mention_for_one_hop(question, block_list)
                        row['relation'] = relation
                        row['mention_labels'] = None if len(mention_labels) == 0 else mention_labels
                    else:
                        row['n_hop'] = '多跳问句'
                        row['relation'] = None
                        row['mention_labels'] = None

                if len(line) == 0:
                    continue

        json_f.close()
    return


if __name__ == '__main__':
    main()
