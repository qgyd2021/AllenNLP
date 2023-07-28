#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Dict, Iterable, List, Union
import logging
import json
from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter

logger = logging.getLogger(__name__)


@DatasetReader.register("hierarchical_classification_json")
class HierarchicalClassificationJsonReader(DatasetReader):
    def __init__(self,
                 n_hierarchical: int = 2,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 segment_sentences: bool = False,
                 max_sequence_length: int = None,
                 skip_label_indexing: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._n_hierarchical = n_hierarchical
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._segment_sentences = segment_sentences
        self._max_sequence_length = max_sequence_length
        self._skip_label_indexing = skip_label_indexing
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        if self._segment_sentences:
            self._sentence_segmenter = SpacySentenceSplitter()

    @overrides
    def _read(self, file_path) -> Iterable[Instance]:
        with open(cached_path(file_path), "r", encoding='utf-8') as data_file:
            for line in data_file.readlines():
                if not line:
                    continue
                items = json.loads(line)
                text = items["text"]

                labels = [items.get("label{}".format(idx), None) for idx in range(self._n_hierarchical)]
                if all(labels):
                    label = '_'.join(labels)
                else:
                    label = None

                if label is not None:
                    if self._skip_label_indexing:
                        try:
                            label = int(label)
                        except ValueError:
                            raise ValueError('Labels must be integers if skip_label_indexing is True.')
                    else:
                        label = str(label)
                instance = self.text_to_instance(text=text, label=label)
                if instance is not None:
                    yield instance

    def _truncate(self, tokens):
        if len(tokens) > self._max_sequence_length:
            tokens = tokens[:self._max_sequence_length]
        return tokens

    @overrides
    def text_to_instance(self, text: str, label: Union[str, int] = None) -> Instance:
        fields: Dict[str, Field] = {}
        if self._segment_sentences:
            sentences: List[Field] = []
            sentence_splits = self._sentence_segmenter.split_sentences(text)
            for sentence in sentence_splits:
                word_tokens = self._tokenizer.tokenize(sentence)
                if self._max_sequence_length is not None:
                    word_tokens = self._truncate(word_tokens)
                sentences.append(TextField(word_tokens, self._token_indexers))
            fields['tokens'] = ListField(sentences)
        else:
            tokens = self._tokenizer.tokenize(text)
            if self._max_sequence_length is not None:
                tokens = self._truncate(tokens)
            fields['tokens'] = TextField(tokens, self._token_indexers)
        if label is not None:
            fields['label'] = LabelField(label,
                                         skip_indexing=self._skip_label_indexing)
        return Instance(fields)
