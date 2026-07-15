#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import sys
from pathlib import Path

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, "../../../"))

from allennlp.data.data_loaders.multiprocess_data_loader import MultiProcessDataLoader
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.basic_classifier import BasicClassifier
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer
from allennlp.training.optimizers import AdamOptimizer
import torch

from toolbox.allennlp.data.dataset_readers.text_classification_json import TextClassificationJsonReader
from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_path",
        default=(project_path / "pretrained_models/dienstag/chinese-bert-wwm-ext").as_posix(),
        type=str,
    )
    parser.add_argument(
        "--train_subset",
        default="data_dir/train.jsonl",
        type=str,
    )
    parser.add_argument(
        "--valid_subset",
        default="data_dir/valid.jsonl",
        type=str,
    )
    parser.add_argument(
        "--vocabulary_dir",
        default="data_dir/vocabulary",
        type=str,
    )
    parser.add_argument(
        "--serialization_dir",
        default="data_dir/serialization_dir",
        type=str,
    )
    parser.add_argument(
        "--max_length",
        default=512,
        type=int,
    )
    parser.add_argument(
        "--embedding_dim",
        default=128,
        type=int,
    )
    parser.add_argument(
        "--num_filters",
        default=128,
        type=int,
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-3,
        type=float,
    )
    parser.add_argument(
        "--token_min_padding_length",
        default=9,
        type=int,
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    vocabulary_dir = Path(args.vocabulary_dir)
    vocabulary_dir.parent.mkdir(parents=True, exist_ok=True)

    dataset_reader = TextClassificationJsonReader(
        token_indexers={
            "tokens": SingleIdTokenIndexer(
                namespace="tokens",
                lowercase_tokens=True,
                token_min_padding_length=args.token_min_padding_length,
            ),
        },
        tokenizer=PretrainedTransformerTokenizer(
            model_name=args.pretrained_model_path,
            max_length=args.max_length,
        ),
    )

    vocabulary = Vocabulary.from_files(args.vocabulary_dir)

    data_loader = MultiProcessDataLoader(
        reader=dataset_reader,
        data_path=args.train_subset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    data_loader.index_with(vocabulary)

    validation_data_loader = MultiProcessDataLoader(
        reader=dataset_reader,
        data_path=args.valid_subset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_data_loader.index_with(vocabulary)

    model = BasicClassifier(
        vocab=vocabulary,
        text_field_embedder=BasicTextFieldEmbedder(
            token_embedders={
                "tokens": Embedding(
                    num_embeddings=vocabulary.get_vocab_size("tokens"),
                    embedding_dim=args.embedding_dim,
                ),
            }
        ),
        seq2vec_encoder=CnnEncoder(
            embedding_dim=args.embedding_dim,
            num_filters=args.num_filters,
            ngram_filter_sizes=(1, 2, 3, 4, 5, 6, 7, 8, 9),
        ),
    )

    parameters = [(n, v) for n, v in model.named_parameters()]

    optimizer = AdamOptimizer(
        model_parameters=parameters,
        lr=args.learning_rate,
    )

    if torch.cuda.is_available():
        cuda_device = 0
        model.cuda(device=0)
    else:
        cuda_device = -1

    print(cuda_device)

    trainer = GradientDescentTrainer(
        cuda_device=cuda_device,
        model=model,
        optimizer=optimizer,
        checkpointer=Checkpointer(
            serialization_dir=args.serialization_dir,
            keep_most_recent_by_count=10,
        ),
        data_loader=data_loader,
        validation_data_loader=validation_data_loader,
        patience=5,
        validation_metric="+accuracy",
        num_epochs=100,
        serialization_dir=args.serialization_dir,
    )
    trainer.train()
    return


if __name__ == "__main__":
    main()
