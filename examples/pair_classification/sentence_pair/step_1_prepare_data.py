#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://huggingface.co/datasets/qgyd2021/sentence_pair
"""
import argparse
import json
import os
from pathlib import Path

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

from datasets import load_dataset
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", default="qgyd2021/sentence_pair", type=str)
    parser.add_argument("--dataset_split", default=None, type=str)
    parser.add_argument(
        "--dataset_cache_dir",
        default=(project_path / "hub_datasets").as_posix(),
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
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    train_subset = Path(args.train_subset)
    valid_subset = Path(args.valid_subset)

    train_subset.parent.mkdir(parents=True, exist_ok=True)
    valid_subset.parent.mkdir(parents=True, exist_ok=True)

    train_f = open(args.train_subset, "w", encoding="utf-8")
    valid_f = open(args.valid_subset, "w", encoding="utf-8")

    names = ["diac2019", "afqmc", "ccks2018_task3", "chinese_sts", 
             "chip2019", "covid_19", "lcqmc", "pawsx_zh", "sts_b"]

    for subset in tqdm(names):
        dataset_dict = load_dataset(
            path=args.dataset_path,
            name=subset,
            split=args.dataset_split,
            cache_dir=args.dataset_cache_dir,
        )
        train_dataset = dataset_dict["train"]
        valid_dataset = dataset_dict["validation"]
        for sample in train_dataset:
            row = {
                "text1": sample["sentence1"],
                "text2": sample["sentence2"],
                "label": sample["label"],
            }
            row = json.dumps(row, ensure_ascii=False)
            train_f.write("{}\n".format(row))

        for sample in valid_dataset:
            row = {
                "text1": sample["sentence1"],
                "text2": sample["sentence2"],
                "label": sample["label"],
            }
            row = json.dumps(row, ensure_ascii=False)
            valid_f.write("{}\n".format(row))

    train_f.close()
    valid_f.close()
    return


if __name__ == "__main__":
    main()
