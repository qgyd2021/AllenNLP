#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import os

from allennlp.data.vocabulary import Vocabulary
import torch

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_path",
        default=(project_path / "pretrained_models/chinese-bert-wwm-ext").as_posix(),
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
    parser.add_argument(
        "--vocabulary_dir",
        default="data_dir/vocabulary",
        type=str
    )
    parser.add_argument(
        "--serialization_dir",
        default="data_dir/serialization_dir",
        type=str
    )
    parser.add_argument(
        "--json_config_dir",
        default="data_dir",
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    vocabulary = Vocabulary.from_files(args.vocabulary_dir)

    if torch.cuda.is_available():
        cuda_device = 0
    else:
        cuda_device = -1

    json_config = {
        "dataset_reader": {
            "type": "pair_classification_json",
            "tokenizer": {
                "type": "pretrained_transformer",
                "model_name": args.pretrained_model_path
            },
            "token_indexers": {
                "tokens": {
                    "type": "pretrained_transformer",
                    "model_name": args.pretrained_model_path
                },
            }
        },
        "train_data_path": args.train_subset,
        "validation_data_path": args.valid_subset,
        "vocabulary": {
            "directory_path": args.vocabulary_dir,
        },
        "model": {
            "type": "decomposable_attention",
            "text_field_embedder": {
                "token_embedders": {
                    "tokens": {
                        "type": "pretrained_transformer",
                        "model_name": args.pretrained_model_path,
                        "train_parameters": True
                    }
                }
            },
            "attend_feedforward": {
                "input_dim": 768,
                "num_layers": 2,
                "hidden_dims": 384,
                "activations": "relu",
                "dropout": 0.1
            },
            "matrix_attention": {
                "type": "dot_product"
            },
            "compare_feedforward": {
                "input_dim": 1536,
                "num_layers": 2,
                "hidden_dims": 768,
                "activations": "relu",
                "dropout": 0.1
            },
            "aggregate_feedforward": {
                "input_dim": 1536,
                "num_layers": 2,
                "hidden_dims": [768, vocabulary.get_vocab_size(namespace="labels")],
                "activations": "relu",
                "dropout": 0.1
            }
        },
        "data_loader": {
            "type": "multiprocess",
            "batch_size": 64,
            "shuffle": True
        },
        "trainer": {
            "type": "gradient_descent",
            "cuda_device": cuda_device,
            "optimizer": {
                "type": "bert_adam",
                "lr": 5e-5,
                "warmup": 0.1,
                "t_total": 200000,
                "schedule": "warmup_linear"
            },
            "checkpointer": {
                "serialization_dir": args.serialization_dir,
                "keep_most_recent_by_count": 10
            },
            "patience": 5,
            "validation_metric": "+accuracy",
            "num_epochs": 20
        }
    }

    with open(os.path.join(args.json_config_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(json_config, f, indent=4, ensure_ascii=False)
    return


if __name__ == '__main__':
    main()
