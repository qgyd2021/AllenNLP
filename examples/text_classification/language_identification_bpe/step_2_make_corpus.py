#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, "../../"))

import sentencepiece as spm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus_file",
        default="data_dir/train.jsonl",
        type=str
    )
    parser.add_argument(
        "--output_file",
        default="bpe_corpus.txt",
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    with open(args.corpus_file, "r", encoding="utf-8") as fin, open(args.output_file, "w", encoding="utf-8") as fout:
        for row in fin:
            row = json.loads(row)
            text = row["text"]

            fout.write("{}\n".format(text))
    return


if __name__ == "__main__":
    main()
