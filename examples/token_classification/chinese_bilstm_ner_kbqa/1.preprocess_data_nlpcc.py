#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
from typing import List, Union

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default=(project_path / "examples/token_classification/chinese_bilstm_ner_kbqa/data_dir/NLPCC-KBQA/nlpcc2016-2018.kbqa.train").as_posix(),
        type=str
    )
    parser.add_argument(
        "--pre_train_subset",
        default="data_dir/kbqa-train.json",
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


def main():
    args = get_args()

    json_f = open(args.pre_train_subset, "a+", encoding="utf-8")

    with open(args.data_path, "r", encoding="utf-8") as f:
        question_id: str = None
        question: str = None
        sbv: str = None
        rel: str = None
        obv: str = None

        for line in f:
            line = str(line).strip()
            # print(line)

            if line.startswith('<triple id'):

                _, line = line.split('>	')
                line = line.split('|||')
                if len(line) != 3:
                    continue
                if len(line[2].strip()) == 0:
                    continue
                sbv = line[0].strip()
                rel = line[1].strip()
                obv = line[2].strip()
                continue

            if line.startswith('<question id'):
                line = line.split('>	')
                question = line[1].strip()
                _, question_id = line[0].split(sep='=', maxsplit=1)
                # print(question_id, question)
                continue
            if line.startswith('==='):

                mention = max_length_common_substring(question, sbv)
                if mention is None:
                    continue
                row = {
                    'source': 'nlpcc2016-2018',
                    'question_id': question_id,
                    'question': question,
                    'sbv': sbv,
                    'rel': rel,
                    'obv': obv,
                    'mention_labels': [mention],
                }
                row = json.dumps(row, ensure_ascii=False)
                json_f.write('{}\n'.format(row))

    return


if __name__ == '__main__':
    main()
