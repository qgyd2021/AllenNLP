#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://modelscope.cn/models/dienstag/chinese-bert-wwm-ext
"""
import argparse
import os
import sys
from pathlib import Path

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, "../../../"))

from project_settings import project_path
from toolbox.modelscope.download import snapshot_download


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        default="dienstag/chinese-bert-wwm-ext",
        type=str,
    )
    parser.add_argument(
        "--revision",
        default="master",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default=(project_path / "pretrained_models/dienstag/chinese-bert-wwm-ext").as_posix(),
        type=str,
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    model_dir = snapshot_download(
        model_id=args.model_id,
        local_dir=Path(args.output_dir),
        revision=args.revision,
    )

    print("Model downloaded to {}".format(model_dir.as_posix()))
    return


if __name__ == "__main__":
    main()
