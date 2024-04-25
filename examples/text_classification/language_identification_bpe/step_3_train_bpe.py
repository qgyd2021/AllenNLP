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
        default="bpe_corpus.txt",
        type=str
    )
    parser.add_argument(
        "--model_prefix",
        default="data_dir/tokenizer",
        type=str
    )
    parser.add_argument(
        "--vocab_size",
        default=50000,
        type=int
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    spm.SentencePieceTrainer.Train(
        input=args.corpus_file,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=1.0,
        model_type="bpe",
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_id=0,
        unk_piece="[UNK]",
        bos_piece="[BOS]",
        eos_piece="[EOS]",
        pad_piece="[PAD]",
    )
    return


if __name__ == "__main__":
    main()
