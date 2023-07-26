#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
from allennlp_models.tagging.dataset_readers.ontonotes_ner import OntonotesNamedEntityRecognition
from allennlp_models.common.ontonotes import Ontonotes

allennlp_models 有专门读取 Connll2012 数据集的工具,
但是, 使用这个工具, 它会默认使用数据集中的分词.
则这就不能转化到 bert 模型的分词作训练, 因此, 需要转化.

"""
import argparse
import copy
import json
from pathlib import Path
from typing import List

from allennlp_models.common.ontonotes import Ontonotes
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.tokenizers.token_class import Token
from tqdm import tqdm

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_path",
        default=(project_path / "pretrained_models/chinese-bert-wwm-ext").as_posix(),
        type=str
    )
    parser.add_argument(
        "--file_dir",
        default="data_dir/conll-2012/v4/data/train",
        type=str
    )
    parser.add_argument(
        "--output_file",
        default="train.json",
        type=str
    )
    parser.add_argument(
        "--domain_identifier",
        default="data/chinese/annotations",
        type=str
    )
    args = parser.parse_args()
    return args


ontonotes_reader = Ontonotes()


def to_char_level_tags(words: List[str], pos_tags: List[str]):
    result_words = list()
    result_pos_tags = list()
    for word, pos_tag in zip(words, pos_tags):

        for idx, char in enumerate(list(word)):
            if idx == 0:
                result_words.append(char)
                result_pos_tags.append('B-{}'.format(pos_tag))
            else:
                result_words.append(char)
                result_pos_tags.append('I-{}'.format(pos_tag))

    return result_words, result_pos_tags


def char_level_tags_convert(tokenizer: PretrainedTransformerTokenizer,
                            char_words: List[str], char_pos_tags: List[str]):
    sentence = ''.join(char_words)
    tokens: List[Token] = tokenizer.tokenize(sentence)
    tokens = [token.text for token in tokens]
    tokens_ = copy.deepcopy(tokens)

    if len(tokens) - 2 == len(char_words):
        result_tokens = tokens
        result_named_entities = ['O'] + char_pos_tags + ['O']
    else:
        result_tokens = list()
        result_named_entities = list()

        cls = tokens_.pop(0)
        result_tokens.append(cls)
        result_named_entities.append('O')

        target_token: str = None
        search_token: str = None
        for char_word, char_named_entity in zip(char_words, char_pos_tags):
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


def convert(tokenizer: PretrainedTransformerTokenizer,
            file_dir: str, to_filename: str, domain_identifier: str = None):

    with open(to_filename, 'w', encoding='utf-8') as f:
        for conll_file in tqdm(ontonotes_reader.dataset_path_iterator(file_dir)):
            conll_file = Path(conll_file).as_posix()
            if domain_identifier is not None and not conll_file.__contains__(domain_identifier):
                continue
            for sentence in ontonotes_reader.sentence_iterator(conll_file):
                char_words, char_pos_tags = to_char_level_tags(sentence.words, sentence.pos_tags)
                sentence = ''.join(char_words)

                try:
                    words, pos_tags = char_level_tags_convert(tokenizer, char_words, char_pos_tags)
                except Exception as e:
                    continue

                row = {
                    'sentence': sentence,
                    'tokens': words,
                    'tags': pos_tags,
                }
                row = json.dumps(row, ensure_ascii=False)
                f.write('{}\n'.format(row))

    return file_dir


def main():
    args = get_args()

    tokenizer = PretrainedTransformerTokenizer(
        model_name=args.pretrained_model_path
    )

    convert(
        tokenizer=tokenizer,
        file_dir=args.file_dir,
        to_filename=args.output_file,
        domain_identifier=args.domain_identifier
    )
    return


if __name__ == '__main__':
    main()
