#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import time

from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.archival import archive_model, load_archive
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.predictors.predictor import Predictor
from allennlp.models.model import Model
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp_models.tagging.models.crf_tagger import CrfTagger
import torch

from project_settings import project_path
from toolbox.allennlp_models.tagging.dataset_readers.bio_tagging_json_reader import BioTaggingJsonReader
from toolbox.allennlp_models.tagging.predictors.sentence_tagger_with_tokenizer import SentenceTaggerWithTokenizerPredictor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        default="给我推荐一些篮球游戏？",
        type=str
    )
    parser.add_argument(
        "--archive_file",
        default=(project_path / "trained_models/chinese_lstm_pos_conll2012").as_posix(),
        type=str
    )
    parser.add_argument(
        "--pretrained_model_path",
        default=(project_path / "pretrained_models/chinese-bert-wwm-ext").as_posix(),
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    archive = load_archive(archive_file=args.archive_file)

    tokenizer = PretrainedTransformerTokenizer(
        model_name=args.pretrained_model_path
    )
    predictor = SentenceTaggerWithTokenizerPredictor(
        model=archive.model,
        dataset_reader=archive.dataset_reader,
        tokenizer=tokenizer

    )

    json_dict = {
        "sentence": args.text
    }

    begin_time = time.time()
    outputs = predictor.predict_json(
        json_dict
    )
    tags = outputs['tags']
    print(tags)

    predicted_spans = predictor.bio_decode(outputs)
    print(predicted_spans)
    print('time cost: {}'.format(time.time() - begin_time))

    return


if __name__ == '__main__':
    main()
