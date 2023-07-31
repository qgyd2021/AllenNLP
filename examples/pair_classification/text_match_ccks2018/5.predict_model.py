#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import time

from allennlp.data.data_loaders.multiprocess_data_loader import MultiProcessDataLoader
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.modules.token_embedders.pretrained_transformer_embedder import PretrainedTransformerEmbedder
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import FeedForward
from allennlp.modules.matrix_attention import DotProductMatrixAttention
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp_models.pair_classification.models.decomposable_attention import DecomposableAttention
from allennlp_models.pair_classification.predictors.textual_entailment import TextualEntailmentPredictor

import torch

from project_settings import project_path
from toolbox.allennlp_models.pair_classification.dataset_readers.pair_classification_json import PairClassificationJsonReader


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
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    dataset_reader = PairClassificationJsonReader(
        tokenizer=PretrainedTransformerTokenizer(
            model_name=args.pretrained_model_path,
        ),
        token_indexers={
            "tokens": PretrainedTransformerIndexer(
                model_name=args.pretrained_model_path,
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

    model = DecomposableAttention(
        vocab=vocabulary,
        text_field_embedder=BasicTextFieldEmbedder(
            token_embedders={
                'tokens': PretrainedTransformerEmbedder(
                    model_name=args.pretrained_model_path,
                    train_parameters=True,
                )
            }
        ),
        attend_feedforward=FeedForward(
            input_dim=768,
            num_layers=2,
            hidden_dims=384,
            activations=torch.nn.ReLU(),
            dropout=0.1,
        ),
        matrix_attention=DotProductMatrixAttention(),
        compare_feedforward=FeedForward(
            input_dim=1536,
            num_layers=2,
            hidden_dims=768,
            activations=torch.nn.ReLU(),
            dropout=0.1,
        ),
        aggregate_feedforward=FeedForward(
            input_dim=1536,
            num_layers=2,
            hidden_dims=[1536, 2],
            activations=torch.nn.ReLU(),
            dropout=0.1,
        ),
    )

    checkpoint_path = os.path.join(args.serialization_dir, "best.th")
    with open(checkpoint_path, 'rb') as f:
        state_dict = torch.load(f, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    predictor = TextualEntailmentPredictor(
        model=model,
        dataset_reader=dataset_reader,
    )

    while True:
        text1 = input("text1: ")
        if text1 == "Quit":
            break
        text2 = input("text2: ")
        if text2 == "Quit":
            break

        json_dict = {"premise": text1, "hypothesis": text2}

        begin_time = time.time()
        outputs = predictor.predict_json(
            json_dict
        )
        label = outputs["label"]
        print(label)

        print('time cost: {}'.format(time.time() - begin_time))
    return


if __name__ == '__main__':
    main()
