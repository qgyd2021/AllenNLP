#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import copy
from itertools import chain
import os
from pathlib import Path
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, "../../../"))

from allennlp.data.data_loaders.multiprocess_data_loader import MultiProcessDataLoader
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.basic_classifier import BasicClassifier
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders.gated_cnn_encoder import GatedCnnEncoder
from allennlp.modules.seq2seq_encoders.pytorch_transformer_wrapper import PytorchTransformer
from allennlp.modules.seq2seq_encoders.pass_through_encoder import PassThroughEncoder
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.modules.seq2vec_encoders.boe_encoder import BagOfEmbeddingsEncoder
from allennlp.modules.seq2vec_encoders.cls_pooler import ClsPooler
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer
from allennlp.training.optimizers import AdamOptimizer
from pytorch_pretrained_bert.optimization import BertAdam
import torch

from toolbox.allennlp.data.dataset_readers.text_classification_json import TextClassificationJsonReader
from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_path",
        default=(project_path / "pretrained_models/bert-base-multilingual-cased").as_posix(),
        type=str
    )
    parser.add_argument(
        "--train_subset",
        default="file_dir/train.jsonl",
        type=str
    )
    parser.add_argument(
        "--valid_subset",
        default="file_dir/valid.jsonl",
        type=str
    )
    parser.add_argument(
        "--vocabulary_dir",
        default="file_dir/vocabulary",
        type=str
    )
    parser.add_argument(
        "--serialization_dir",
        default="file_dir/serialization_dir",
        type=str
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
                token_min_padding_length=5,
            ),
        },
        tokenizer=PretrainedTransformerTokenizer(
            model_name=args.pretrained_model_path,
        ),
        max_sequence_length=256,
    )

    vocabulary = Vocabulary.from_files(args.vocabulary_dir)

    data_loader = MultiProcessDataLoader(
        reader=dataset_reader,
        data_path=args.train_subset,
        batch_size=64,
        shuffle=True,
        # max_instances_in_memory=64 * 5
    )
    data_loader.index_with(vocabulary)

    validation_data_loader = MultiProcessDataLoader(
        reader=dataset_reader,
        data_path=args.valid_subset,
        batch_size=64,
        shuffle=True,
        # max_instances_in_memory=64 * 5
    )
    validation_data_loader.index_with(vocabulary)

    model = BasicClassifier(
        vocab=vocabulary,
        text_field_embedder=BasicTextFieldEmbedder(
            token_embedders={
                "tokens": Embedding(
                    embedding_dim=64,
                    num_embeddings=vocabulary.get_vocab_size("tokens")
                ),
            }
        ),
        seq2seq_encoder=PytorchTransformer(
            input_dim=64,
            num_layers=4,
            feedforward_hidden_dim=128,
            num_attention_heads=4,
            positional_encoding="sinusoidal",
            positional_embedding_size=512,
            dropout_prob=0.5,
        ),
        seq2vec_encoder=ClsPooler(
            embedding_dim=64,
        )
    )

    # with open('serialization_dir/best.th', 'rb') as f:
    #     state_dict = torch.load(f, map_location=torch.device('cpu'))
    # model.load_state_dict(state_dict)
    # model.train()

    # parameters = [v for n, v in model.named_parameters()]
    # optimizer = BertAdam(
    #     params=parameters,
    #     lr=5e-5,
    #     warmup=0.1,
    #     # t_total=50000,
    #     t_total=200000,
    #     schedule='warmup_linear'
    # )

    parameters = [(n, v) for n, v in model.named_parameters()]
    optimizer = AdamOptimizer(
        model_parameters=parameters,
        lr=1e-4,
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
        validation_metric='+accuracy',
        num_epochs=100,
        serialization_dir=args.serialization_dir,
    )
    trainer.train()
    return


if __name__ == '__main__':
    main()
