#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import copy
from itertools import chain
import os
from pathlib import Path

from allennlp.data.data_loaders.multiprocess_data_loader import MultiProcessDataLoader
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.basic_classifier import BasicClassifier
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders.gated_cnn_encoder import GatedCnnEncoder
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer
from allennlp.training.checkpointer import Checkpointer
from pytorch_pretrained_bert.optimization import BertAdam
from allennlp.training.optimizers import AdamOptimizer
import torch

from toolbox.allennlp.data.dataset_readers.text_classification_json import TextClassificationJsonReader
from toolbox.allennlp.data.tokenizers.sentence_piece_bpe_tokenizer import SentencePieceBPETokenizer
from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bpe_tokenizer_file",
        default="data_dir/bpe_tokenizer.model",
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
    parser.add_argument(
        "--vocabulary_dir",
        default="vocabulary",
        type=str
    )
    parser.add_argument(
        "--serialization_dir",
        default="serialization_dir",
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
        tokenizer=SentencePieceBPETokenizer(
            bpe_tokenizer_file=args.bpe_tokenizer_file
        ),
    )

    vocabulary = Vocabulary.from_files(args.vocabulary_dir)

    data_loader = MultiProcessDataLoader(
        reader=dataset_reader,
        data_path=args.train_subset,
        batch_size=64,
        shuffle=True,
    )
    data_loader.index_with(vocabulary)

    validation_data_loader = MultiProcessDataLoader(
        reader=dataset_reader,
        data_path=args.valid_subset,
        batch_size=64,
        shuffle=True,
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
        seq2seq_encoder=GatedCnnEncoder(
            input_dim=64,
            layers=[[[4, 64]], [[4, 64], [4, 64]], [[4, 64]]],
            dropout=0.05,
        ),
        seq2vec_encoder=CnnEncoder(
            embedding_dim=128,
            num_filters=64,
            ngram_filter_sizes=(2, 3, 4, 5),
            output_dim=64,
        )
    )

    # with open('serialization_dir/best.th', 'rb') as f:
    #     state_dict = torch.load(f, map_location=torch.device('cpu'))
    # model.load_state_dict(state_dict)
    model.train()

    parameters = [v for n, v in model.named_parameters()]

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
