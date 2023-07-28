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
from allennlp_models.rc.modules.seq2seq_encoders.stacked_self_attention import StackedSelfAttentionEncoder
from allennlp.predictors.text_classifier import TextClassifierPredictor
import torch

from project_settings import project_path
from toolbox.allennlp_models.text_classifier.models.hierarchical_text_classifier import HierarchicalClassifier
from toolbox.allennlp_models.text_classifier.dataset_readers.hierarchical_classification_json import HierarchicalClassificationJsonReader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_path",
        default=(project_path / "pretrained_models/chinese-bert-wwm-ext").as_posix(),
        type=str
    )
    parser.add_argument('--hierarchical_labels_pkl', default='data_dir/hierarchical_labels.pkl', type=str)
    parser.add_argument('--vocabulary_dir', default='data_dir/vocabulary', type=str)

    parser.add_argument(
        "--serialization_dir",
        default="data_dir/serialization_dir2",
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    dataset_reader = HierarchicalClassificationJsonReader(
        token_indexers={
            'tokens': SingleIdTokenIndexer(
                namespace='tokens',
                lowercase_tokens=True,
                token_min_padding_length=5,
            )
        },
        tokenizer=PretrainedTransformerTokenizer(
            model_name=os.path.join(project_path, args.pretrained_model_path),
        ),
    )

    vocabulary = Vocabulary.from_files(args.vocabulary_dir)

    model = HierarchicalClassifier(
        vocab=vocabulary,
        hierarchical_labels_pkl=args.hierarchical_labels_pkl,
        text_field_embedder=BasicTextFieldEmbedder(
            token_embedders={
                'tokens': Embedding(
                    num_embeddings=vocabulary.get_vocab_size('tokens'),
                    embedding_dim=128,
                )
            }
        ),
        seq2seq_encoder=StackedSelfAttentionEncoder(
            input_dim=128,
            hidden_dim=128,
            projection_dim=128,
            feedforward_hidden_dim=128,
            num_layers=2,
            num_attention_heads=4,
            use_positional_encoding=False,
        ),
        seq2vec_encoder=CnnEncoder(
            embedding_dim=128,
            num_filters=32,
            ngram_filter_sizes=(2, 3, 4, 5),
        ),
    )

    checkpoint_path = os.path.join(args.serialization_dir, "best.th")
    with open(checkpoint_path, 'rb') as f:
        state_dict = torch.load(f, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    predictor = TextClassifierPredictor(
        model=model,
        dataset_reader=dataset_reader,
    )

    while True:
        text = input("text: ")
        if text == "Quit":
            break

        json_dict = {'sentence': text}

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
