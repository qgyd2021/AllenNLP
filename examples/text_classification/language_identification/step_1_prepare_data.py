#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import random
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
|   af   |    afrikaans   |   35214   |       spc        |
|   ar   |     arabic     |  100000   |     iwslt2017    |
|   bg   |   bulgarian    |  100000   |       xnli       |
|   bn   |    bengali     |  36064    |  open_subtitles  |
|   bs   |    bosnian     |  10212    |  open_subtitles  |
|   cs   |     czech      |  100000   |       emea       |
|   da   |     danish     |  100000   |  open_subtitles  |
|   de   |     german     |  100000   |     iwslt2017    |
|   el   |  modern greek  |  100000   |       emea       |
|   en   |    english     |  200000   |     iwslt2017    |
|   eo   |   esperanto    |  94101    | tatoeba; open_subtitles |
|   es   |    spanish     |  100000   |       xnli       |
|   et   |    estonian    |  100000   |       emea       |
|   fi   |    finnish     |  100000   |    ecb; kde4     |
|   fo   |    faroese     |  23807    |  nordic_langid   |
|   fr   |    french      |  100000   |     iwslt2017    |
|   ga   |    irish       |  100000   | multi_para_crawl |
|   gl   |    galician    |  3096     |     tatoeba      |
|   hi   |     hindi      |  100000   |       xnli       |
|  hi_en |     hindi      |  7180     | cmu_hinglish_dog |
|   hr   |    croatian    |  95844    |   hrenwac_para   |
|   hu   |    hungarian   |  3801     |   europa_ecdc_tm; europa_eac_tm   |
|   hy   |    armenian    |   660     |  open_subtitles  |
|   id   |   indonesian   |  23940    |   id_panl_bppt   |
|   is   |   icelandic    |  100000   | multi_para_crawl |
|   it   |    italian     |  100000   |     iwslt2017    |
|   ja   |    japanese    |  100000   |     iwslt2017    |
|   ko   |    korean      |  100000   |     iwslt2017    |
|   lt   |   lithuanian   |  100000   |       emea       |
|   lv   |    latvian     |  100000   | multi_para_crawl |
|   mr   |    marathi     |  51807    |     tatoeba      |
|   mt   |    maltese     |  100000   | multi_para_crawl |
|   nl   |    dutch       |  100000   |       kde4       |
|   no   |   norwegian    |  100000   | multi_para_crawl |
|   pl   |    polish      |  100000   | para_crawl_en_pl |
|   pt   |   portuguese   |  100000   | para_crawl_en_pt |
|   ro   |    romanian    |  100000   |     iwslt2017    |
|   ru   |    russian     |  100000   |       xnli       |
|   sk   |    slovak      |  100000   | multi_para_crawl |
|   sl   |   slovenian    |  100000   | para_crawl_en_sl |
|   sw   |    swahili     |  100000   |       xnli       |
|   sv   |    swedish     |  100000   |       kde4       |
|   th   |     thai       |  100000   |       xnli       |
|   tl   |    tagalog     |  97241    | multi_para_crawl |
|   tn   |    serpeti     |  100000   |   autshumato     |
|   tr   |    turkish     |  100000   |       xnli       |
|   ts   |    dzonga      |  100000   |    autshumato    |
|   uk   |    ukrainian   |  88533    |  para_pat_en_uk  |
|   ur   |     urdu       |  100000   |       xnli       |
|   vi   |   vietnamese   |  100000   |       xnli       |
|   yo   |     yoruba     |   9970    |    menyo20k_mt   |
|   zh   |    chinese     |  200000   |       xnli       |
|   zu   |  zulu, south africa  |  26801   |    autshumato    |
"""


def main():
    args = get_args()

    subset_dataset_dict = dict()

    lines = s.strip().split("\n")

    language_map = {
        "zh-cn": "zh"
    }

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

            count = 0
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

                    if language in language_map.keys():
                        language = language_map[language]

                    if count > total:
                        break

                    if language != abbr:
                        continue

                    split = "train" if random.random() < 0.8 else "valid"

                    row_ = {
                        "text": text,
                        "label": language,
                        "language": full,
                        "data_source": data_source,
                        "split": split,
                    }
                    row_ = json.dumps(row_, ensure_ascii=False)

                    if split == "train":
                        ftrain.write("{}\n".format(row_))
                    elif split == "valid":
                        fvalid.write("{}\n".format(row_))
                    else:
                        raise AssertionError

                    count += 1

    return


if __name__ == "__main__":
    main()
