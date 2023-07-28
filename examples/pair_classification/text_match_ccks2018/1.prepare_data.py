#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import random


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus_train_subset",
        default="data_dir/train.txt",
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
    args = parser.parse_args()
    return args


def main():
    """
    entailment
    contradiction
    """
    args = get_args()

    train_f = open(args.train_subset, "w", encoding="utf-8")
    valid_f = open(args.valid_subset, "w", encoding="utf-8")

    with open(args.corpus_train_subset, "r", encoding="utf-8") as f:
        for row in f:
            row = str(row).strip()
            row = row.split("\t")
            if len(row) != 3:
                continue
            text1, text2, label = row

            if label in (0, "0"):
                label = "contradiction"
            else:
                label = "entailment"
            row = {
                "text1": text1,
                "text2": text2,
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


if __name__ == '__main__':
    main()
