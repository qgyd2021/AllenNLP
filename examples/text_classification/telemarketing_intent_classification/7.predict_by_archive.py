#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import time

from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.models.archival import archive_model, load_archive
from allennlp_models.rc.modules.seq2seq_encoders.stacked_self_attention import StackedSelfAttentionEncoder
from allennlp.predictors.predictor import Predictor
from allennlp.predictors.text_classifier import TextClassifierPredictor
import torch

from project_settings import project_path
from toolbox.allennlp_models.text_classifier.models.hierarchical_text_classifier import HierarchicalClassifier
from toolbox.allennlp_models.text_classifier.dataset_readers.hierarchical_classification_json import HierarchicalClassificationJsonReader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        default="给我推荐一些篮球游戏？",
        type=str
    )
    parser.add_argument(
        "--archive_file",
        default=(project_path / "trained_models/telemarketing_intent_classification_vi").as_posix(),
        type=str
    )
    parser.add_argument(
        "--pretrained_model_path",
        default=(project_path / "pretrained_models/chinese-bert-wwm-ext").as_posix(),
        type=str
    )
    parser.add_argument(
        "--predictor_name",
        default="text_classifier",
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    archive = load_archive(archive_file=args.archive_file)

    predictor = Predictor.from_archive(archive, predictor_name=args.predictor_name)

    json_dict = {
        "sentence": args.text
    }

    begin_time = time.time()
    outputs = predictor.predict_json(
        json_dict
    )
    outputs = predictor._model.decode(outputs)
    label = outputs['label']
    print(label)
    print('time cost: {}'.format(time.time() - begin_time))

    return


if __name__ == '__main__':
    main()
