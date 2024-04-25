#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, "../../"))

from datasets import load_dataset, DownloadMode

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_cache_dir",
        default=(project_path / "hub_datasets").as_posix(),
        type=str
    )
    parser.add_argument(
        "--train_subset",
        default="train.jsonl",
        type=str
    )
    parser.add_argument(
        "--valid_subset",
        default="valid.jsonl",
        type=str
    )
    args = parser.parse_args()
    return args


s = """
|   ar   |     arabic     |  10000   |     iwslt2017    |
|   bg   |   bulgarian    |  10000   |       xnli       |
|   bn   |    bengali     |  10000   |  open_subtitles  |
|   bs   |    bosnian     |  10000   |  open_subtitles  |
|   cs   |     czech      |  10000   |       ecb        |
|   da   |     danish     |  10000   |  open_subtitles  |
|   de   |     german     |  10000   |       ecb        |
|   el   |  modern greek  |  10000   |       ecb        |
|   en   |    english     |  10000   |       ecb        |
|   eo   |   esperanto    |  10000   |     tatoeba      |
|   es   |    spanish     |  10000   |     tatoeba      |
|   et   |    estonian    |  10000   |      emea        |
|   fi   |    finnish     |  10000   |       ecb        |
|   fo   |    faroese     |  10000   |  nordic_langid   |
|   fr   |    french      |  10000   |     iwslt2017    |
|   ga   |    irish       |  10000   | multi_para_crawl |
|   gl   |    galician    |  3096    |     tatoeba      |
|   hi   |     hindi      |  10000   |  open_subtitles  |
|  hi_en |     hindi      |  7180    | cmu_hinglish_dog |
|   hr   |    croatian    |  10000   |   hrenwac_para   |
|   hu   |    hungarian   |  3801    |   europa_ecdc_tm; europa_eac_tm   |
|   hy   |    armenian    |   660    |  open_subtitles  |
|   id   |   indonesian   |  10000   |   id_panl_bppt   |
|   is   |   icelandic    |  2973    |   europa_ecdc_tm; europa_eac_tm   |
|   it   |    italian     |  10000   |     iwslt2017    |
|   ja   |    japanese    |  10000   |     iwslt2017    |
|   ko   |    korean      |  10000   |     iwslt2017    |
|   lt   |   lithuanian   |  10000   |       emea       |
|   lv   |    latvian     |  4595    |   europa_ecdc_tm; europa_eac_tm   |
|   mr   |    marathi     |  10000   |     tatoeba      |
|   mt   |    maltese     |  10000   | multi_para_crawl |
|   nl   |    dutch       |  10000   |       kde4       |
|   no   |   norwegian    |  10000   | multi_para_crawl |
|   pl   |    polish      |  10000   |       ecb        |
|   pt   |   portuguese   |  10000   |     tatoeba      |
|   ro   |    romanian    |  10000   |       kde4       |
|   ru   |    russian     |  10000   |       xnli       |
|   sk   |    slovak      |  10000   | multi_para_crawl |
|   sl   |   slovenian    |  4589    |   europa_ecdc_tm; europa_eac_tm   |
|   sw   |    swahili     |  10000   |       xnli       |
|   sv   |    swedish     |  10000   |       kde4       |
|   th   |     thai       |  10000   |       xnli       |
|   tl   |    tagalog     |  10000   | multi_para_crawl |
|   tn   |    serpeti     |  10000   |   autshumato     |
|   tr   |    turkish     |  10000   |       xnli       |
|   ts   |    dzonga      |  10000   |    autshumato    |
|   ur   |     urdu       |  10000   |       xnli       |
|   vi   |   vietnamese   |  10000   |       xnli       |
|   yo   |     yoruba     |  9970    |    menyo20k_mt   |
|   zh   |    chinese     |  10000   |       xnli       |
|   zu   |  zulu, south africa  |  10000   |    autshumato    |
"""


def main():
    args = get_args()

    subset_dataset_dict = dict()

    lines = s.strip().split("\n")

    with open(args.train_subset, "w", encoding="utf-8") as ftrain, open(args.valid_subset, "w", encoding="utf-8") as fvalid:
        for line in lines:
            row = str(line).split("|")
            row = [col.strip() for col in row if len(col) != 0]

            if len(row) != 4:
                raise AssertionError("not 4 item, line: {}".format(line))

            abbr = row[0]
            full = row[1]
            total = int(row[2])
            subsets = [e.strip() for e in row[3].split(";")]

            train_count = 0
            valid_count = 0
            for subset in subsets:
                if subset in subset_dataset_dict.keys():
                    dataset_dict = subset_dataset_dict[subset]
                else:
                    dataset_dict = load_dataset(
                        "qgyd2021/language_identification",
                        name=subset,
                        cache_dir=args.dataset_cache_dir,
                        # download_mode=DownloadMode.FORCE_REDOWNLOAD
                    )
                    subset_dataset_dict[subset] = dataset_dict

                train_dataset = dataset_dict["train"]
                for sample in train_dataset:
                    text = sample["text"]
                    language = sample["language"]
                    data_source = sample["data_source"]

                    if train_count > total:
                        break
                    if language == abbr:
                        row_ = {
                            "text": text,
                            "label": language,
                            "data_source": data_source,
                            "split": "train",
                        }
                        row_ = json.dumps(row_, ensure_ascii=False)
                        ftrain.write("{}\n".format(row_))
                        train_count += 1

                if "validation" in dataset_dict:
                    valid_dataset = dataset_dict["validation"]
                    for sample in valid_dataset:
                        text = sample["text"]
                        language = sample["language"]
                        data_source = sample["data_source"]

                        if valid_count > total:
                            break
                        if language == abbr:
                            row_ = {
                                "text": text,
                                "label": language,
                                "data_source": data_source,
                                "split": "valid",
                            }
                            row_ = json.dumps(row_, ensure_ascii=False)
                            fvalid.write("{}\n".format(row_))
                            valid_count += 1

    return


if __name__ == "__main__":
    main()
