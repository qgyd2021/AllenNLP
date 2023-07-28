#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from collections import OrderedDict

from allennlp.data.data_loaders.multiprocess_data_loader import MultiProcessDataLoader
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer
from allennlp.training.checkpointer import Checkpointer
from allennlp_models.tagging.models.crf_tagger import CrfTagger
from pytorch_pretrained_bert.optimization import BertAdam
import torch

from toolbox.allennlp_models.tagging.dataset_readers.bio_tagging_json_reader import BioTaggingJsonReader


def get_args():
    parser = argparse.ArgumentParser()

    # https://huggingface.co/spaces/qgyd2021/ChineseLstmPosConll2012
    parser.add_argument(
        "--pretrained_pos_model_path",
        default=None,
        type=str,
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
        "--train_partial_parameters",
        action="store_true",
    )
    args = parser.parse_args()
    return args


def get_state_dict_by_prefix(state_dict: OrderedDict, prefix: str = None):
    prefix = prefix or ""

    l = len(prefix)
    result = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            k = k[l+1:]
            result[k] = v
    return result


def main():
    args = get_args()

    dataset_reader = BioTaggingJsonReader(
        token_indexers={
            "tokens": SingleIdTokenIndexer(
                namespace="tokens",
                lowercase_tokens=True,
                token_min_padding_length=0,
            )
        }
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

    model = CrfTagger(
        vocab=vocabulary,
        text_field_embedder=BasicTextFieldEmbedder(
            token_embedders={
                "tokens": Embedding(
                    num_embeddings=vocabulary.get_vocab_size(namespace="tokens"),
                    embedding_dim=128
                )
            }
        ),
        encoder=PytorchSeq2SeqWrapper(
            torch.nn.LSTM(
                input_size=128,
                hidden_size=128,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
            )
        ),
        feedforward=FeedForward(
            input_dim=256,
            num_layers=2,
            hidden_dims=256,
            activations=torch.nn.ReLU()
        ),
        label_namespace="labels",
        include_start_end_transitions=True,

    )

    if args.pretrained_pos_model_path is None and args.train_partial_parameters:
        raise AssertionError("train partial parameters only available when pos model given. ")

    if args.pretrained_pos_model_path is not None:
        with open(args.pretrained_pos_model_path, "rb") as f:
            state_dict = torch.load(f, map_location=torch.device("cpu"))

        state_dict1 = get_state_dict_by_prefix(state_dict, prefix="text_field_embedder")
        model.text_field_embedder.load_state_dict(state_dict1, strict=True)

        state_dict2 = get_state_dict_by_prefix(state_dict, prefix="encoder")
        model.encoder.load_state_dict(state_dict2, strict=True)

        state_dict3 = get_state_dict_by_prefix(state_dict, prefix="_feedforward")
        model._feedforward.load_state_dict(state_dict3, strict=True)

    model.train()

    if args.train_partial_parameters:
        # partial parameters
        parameters = list()
        parameters += [v for _, v in model.named_parameters(prefix='tag_projection_layer')]
        parameters += [v for _, v in model.named_parameters(prefix='crf')]
    else:
        # all parameters
        parameters = [v for _, v in model.named_parameters()]

    optimizer = BertAdam(
        params=parameters,
        lr=5e-4,
        warmup=0.1,
        t_total=50000,
        # t_total=200000,
        schedule='warmup_linear'
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
