#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
多层 softmax 实现多极文本分类

由于初始化时, 各层 softmax 的概率趋于平衡.

因此在第一层时 `领域无关` 就分到了 50% 的概率.

`领域相关` 中的各类别去分剩下的 50% 的概率.
这会导致模型一开始时输出的类别全是 `领域无关`, 这导致模型无法优化.

解决方案:
1. 从数据集中去除 `领域无关` 数据. 并训练模型.
2. 等模型收敛之后, 再使用包含 `领域无关` 的数据集, 让模型加载之前的权重, 并重新开始训练模型.

"""
import argparse
import json
import os
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../../'))

from allennlp.data.data_loaders.multiprocess_data_loader import MultiProcessDataLoader
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp_models.rc.modules.seq2seq_encoders.stacked_self_attention import StackedSelfAttentionEncoder
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer
from allennlp.training.checkpointer import Checkpointer
from pytorch_pretrained_bert.optimization import BertAdam
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
    parser.add_argument('--train_subset', default='data_dir/train.json', type=str)
    parser.add_argument('--valid_subset', default='data_dir/valid.json', type=str)
    parser.add_argument("--serialization_dir", default="data_dir/serialization_dir", type=str)
    # parser.add_argument('--checkpoint_path', default="data_dir/serialization_dir/best.th", type=str)
    parser.add_argument('--checkpoint_path', default=None, type=str)

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

    if args.checkpoint_path is not None:
        with open(args.checkpoint_path, "rb") as f:
            state_dict = torch.load(f, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    model.train()

    parameters = [v for n, v in model.named_parameters()]

    optimizer = BertAdam(
        params=parameters,
        lr=5e-4,
        warmup=0.1,
        t_total=10000,
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
