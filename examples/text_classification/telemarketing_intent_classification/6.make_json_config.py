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
    parser.add_argument('--hierarchical_labels_pkl', default='data_dir/hierarchical_labels.pkl', type=str)
    parser.add_argument('--vocabulary_dir', default='data_dir/vocabulary', type=str)
    parser.add_argument('--train_subset', default='data_dir/train.json', type=str)
    parser.add_argument('--valid_subset', default='data_dir/valid.json', type=str)
    parser.add_argument("--serialization_dir", default="data_dir/serialization_dir", type=str)
    parser.add_argument("--json_config_dir", default="data_dir", type=str)
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
            "type": "hierarchical_classification_json",
            "token_indexers": {
                "tokens": {
                    "type": "single_id",
                    "namespace": "tokens",
                    "lowercase_tokens": True,
                    "token_min_padding_length": 5
                }
            },
            "tokenizer": {
                "type": "pretrained_transformer",
                "model_name": args.pretrained_model_path
            }
        },
        "train_data_path": args.train_subset,
        "validation_data_path": args.valid_subset,
        "vocabulary": {
            "directory_path": args.vocabulary_dir,
        },
        "model": {
            "type": "hierarchical_classifier",
            "hierarchical_labels_pkl": args.hierarchical_labels_pkl,
            "text_field_embedder": {
                "token_embedders": {
                    "tokens": {
                        "type": "embedding",
                        "num_embeddings": vocabulary.get_vocab_size(namespace="tokens"),
                        "embedding_dim": 128
                    }
                }
            },
            "seq2seq_encoder": {
                "type": "stacked_self_attention",
                "input_dim": 128,
                "hidden_dim": 128,
                "projection_dim": 128,
                "feedforward_hidden_dim": 128,
                "num_layers": 2,
                "num_attention_heads": 4,
                "use_positional_encoding": False
            },
            "seq2vec_encoder": {
                "type": "cnn",
                "embedding_dim": 128,
                "num_filters": 32,
                "ngram_filter_sizes": (2, 3, 4, 5),
            },
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
                "t_total": 50000,
                "schedule": "warmup_linear"
            },
            "checkpointer": {
                "serialization_dir": args.serialization_dir,
                "keep_most_recent_by_count": 10
            },
            "patience": 5,
            "validation_metric": "+accuracy",
            "num_epochs": 200
        }
    }

    with open(os.path.join(args.json_config_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(json_config, f, indent=4, ensure_ascii=False)
    return


if __name__ == '__main__':
    main()
