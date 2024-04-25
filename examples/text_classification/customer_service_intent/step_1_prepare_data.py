#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import random

from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="user_intent_result.xlsx", type=str)

    parser.add_argument("--train_subset", default="train.jsonl", type=str)
    parser.add_argument("--valid_subset", default="valid.jsonl", type=str)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    label_map = {
        "退换货申请": "退换货申请",
        "退款申请": "退款申请",
        "感谢用语": "感谢用语",
        "招呼用语": "招呼用语",
        "查优惠政策": "查优惠政策",
        "物流问题": "物流问题",
        "查物流政策": "物流问题",
        "查物流信息": "物流问题",
        "未收到商品": "未收到商品",
    }

    train_f = open(args.train_subset, "w", encoding="utf-8")
    valid_f = open(args.valid_subset, "w", encoding="utf-8")

    df = pd.read_excel(args.data_file)
    for i, row in df.iterrows():
        tenant_id = row["tenant_id"]
        text = row["text"]
        language = row["language"]
        intent = row["intent"]
        selected = row["selected"]
        length = row["length"]
        leading_message = row["leading_message"]
        if length > 150:
            continue
        if language != "英语":
            continue
        if leading_message != 1:
            continue
        if selected != 1:
            continue

        if intent in label_map.keys():
            label = "相关领域"
        else:
            label = "无关领域"

        row = {
            "text": text,
            "label": label,
        }
        row = json.dumps(row, ensure_ascii=False)

        flag = random.random()
        if flag < 0.9:
            train_f.write("{}\n".format(row))
        else:
            valid_f.write("{}\n".format(row))

    train_f.close()
    valid_f.close()
    return


if __name__ == '__main__':
    main()
