#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import time

from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.modules.feedforward import FeedForward
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from allennlp.modules.token_embedders.embedding import Embedding
import torch

from allennlp.data.vocabulary import Vocabulary
from allennlp_models.tagging.models.crf_tagger import CrfTagger
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder

from project_settings import project_path
from toolbox.allennlp_models.tagging.dataset_readers.bio_tagging_json_reader import BioTaggingJsonReader
from toolbox.allennlp_models.tagging.predictors.sentence_tagger_with_tokenizer import SentenceTaggerWithTokenizerPredictor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_path",
        default=(project_path / "pretrained_models/chinese-bert-wwm-ext").as_posix(),
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

    tokenizer = PretrainedTransformerTokenizer(
        model_name=args.pretrained_model_path
    )

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

    checkpoint_path = os.path.join(args.serialization_dir, "best.th")
    with open(checkpoint_path, "rb") as f:
        state_dict = torch.load(f, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    predictor = SentenceTaggerWithTokenizerPredictor(
        model=model,
        dataset_reader=dataset_reader,
        tokenizer=tokenizer,
    )

    while True:
        text = input("text: ")
        if text == "Quit":
            break

        # text = '梵蒂冈本应认真的、负责任的检讨历史上有负于中国人民的错误行为，向中国人民道歉。'
        # text = '给我推荐一些篮球游戏？'
        # text = '你们这个要多少利率呢'
        # text = '这就是我们今天要关注的话题。'
        # text = '用于注册FB账号，如果已有FB账号，可略过此项'

        json_dict = {"sentence": text}

        begin_time = time.time()
        outputs = predictor.predict_json(
            json_dict
        )
        tags = outputs["tags"]
        print(tags)

        predicted_spans = predictor.bio_decode(outputs)
        print(predicted_spans)

        print("time cost: {}".format(time.time() - begin_time))
    return


if __name__ == '__main__':
    main()
