#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import random
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, "../../../"))

import pandas as pd

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        default=(project_path / "data/贴子领域分类_模型标注.xlsx").as_posix(),
        type=str,
    )
    parser.add_argument(
        "--train_subset",
        default="data_dir/train.jsonl",
        type=str,
    )
    parser.add_argument(
        "--valid_subset",
        default="data_dir/valid.jsonl",
        type=str,
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    train_f = open(args.train_subset, "w", encoding="utf-8")
    valid_f = open(args.valid_subset, "w", encoding="utf-8")

    df = pd.read_excel(args.data_file)
    for i, row in df.iterrows():
        text = row["文本"]
        label = row["标签"]

        row = {
            "text": text,
            "label": label,
        }
        row = json.dumps(row, ensure_ascii=False)

        if random.random() < 0.9:
            train_f.write("{}\n".format(row))
        else:
            valid_f.write("{}\n".format(row))

    train_f.close()
    valid_f.close()
    return


if __name__ == "__main__":
    main()
