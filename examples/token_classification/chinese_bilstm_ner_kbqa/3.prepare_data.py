#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import copy
import json
from typing import List

from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.tokenizers.token_class import Token

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_path",
        default=(project_path / "pretrained_models/chinese-bert-wwm-ext").as_posix(),
        type=str
    )
    parser.add_argument(
        "--pre_train_subset",
        default="data_dir/kbqa-train.json",
        type=str
    )
    parser.add_argument(
        "--pre_valid_subset",
        default="data_dir/kbqa-valid.json",
        type=str
    )
    parser.add_argument(
        "--pre_test_subset",
        default="data_dir/kbqa-test.json",
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
        "--test_subset",
        default="data_dir/test.json",
        type=str
    )
    args = parser.parse_args()
    return args


def to_char_level_tags(question: str, mentions: List[dict]):
    tokens = list(question)
    tags = ['O'] * len(tokens)

    for mention in mentions:
        begin = mention['begin']
        end = mention['end']

        tags[begin] = 'B'
        for i in range(begin + 1, end):
            tags[i] = 'I'

    return tokens, tags


def char_level_tags_convert(tokenizer: PretrainedTransformerTokenizer, char_words: List[str], char_named_entities: List[str]):
    sentence = ''.join(char_words)
    tokens: List[Token] = tokenizer.tokenize(sentence)
    tokens = [token.text for token in tokens]
    tokens_ = copy.deepcopy(tokens)

    if len(tokens) - 2 == len(char_words):
        result_tokens = tokens
        result_named_entities = ['O'] + char_named_entities + ['O']
    else:
        result_tokens = list()
        result_named_entities = list()

        cls = tokens_.pop(0)
        result_tokens.append(cls)
        result_named_entities.append('O')

        target_token: str = None
        search_token: str = None
        for char_word, char_named_entity in zip(char_words, char_named_entities):
            if target_token is not None:
                if search_token is None:
                    raise AssertionError
                if search_token != target_token:
                    search_token += char_word

                if search_token == target_token:
                    target_token = None
                    search_token = None
                continue

            token: str = tokens_.pop(0)
            if token.startswith('##'):
                token = token[2:]

            if len(char_word) == len(token):
                result_tokens.append(token)
                result_named_entities.append(char_named_entity)
            else:
                if not token.startswith(char_word):
                    raise AssertionError
                result_tokens.append(token)
                result_named_entities.append(char_named_entity)

                target_token = token
                search_token = char_word

        if len(tokens_) != 1:
            raise AssertionError
        sep = tokens_.pop(0)
        result_tokens.append(sep)
        result_named_entities.append('O')

    return result_tokens, result_named_entities


def convert(tokenizer: PretrainedTransformerTokenizer, kbqa_filename: str, to_filename: str):
    with open(to_filename, 'w', encoding='utf-8') as to_file:
        with open(kbqa_filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                mention_labels = line.get('mention_labels')
                question = line.get('question')

                if question is None or mention_labels is None:
                    continue
                # print(question)
                # print(mention_labels)

                char_words, char_named_entities = to_char_level_tags(question, mention_labels)

                try:
                    token_words, token_named_entities = char_level_tags_convert(tokenizer, char_words, char_named_entities)
                except AssertionError as e:
                    continue
                # print(token_words)
                # print(token_named_entities)
                row = {
                    'sentence': question,
                    'tokens': token_words,
                    'tags': token_named_entities,
                }
                row = json.dumps(row, ensure_ascii=False)
                to_file.write('{}\n'.format(row))

    return


def demo1():
    args = get_args()

    tokenizer = PretrainedTransformerTokenizer(
        model_name=args.pretrained_model_path
    )

    convert(
        tokenizer=tokenizer,
        kbqa_filename=args.pre_train_subset,
        to_filename=args.train_subset,
    )
    convert(
        tokenizer=tokenizer,
        kbqa_filename=args.pre_valid_subset,
        to_filename=args.valid_subset
    )
    convert(
        tokenizer=tokenizer,
        kbqa_filename=args.pre_test_subset,
        to_filename=args.test_subset
    )
    return


if __name__ == '__main__':
    demo1()
