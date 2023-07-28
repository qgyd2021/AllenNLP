#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import logging
from typing import Dict, List, Optional, Sequence, Iterable, Union

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, Tokenizer, TokenIndexer, Instance, Field
from allennlp.data.fields import Field, TextField, SequenceLabelField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import WhitespaceTokenizer, Token

logger = logging.getLogger(__name__)


@DatasetReader.register("pair_classification_json")
class PairClassificationJsonReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        label_namespace: str = "labels",
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.label_namespace = label_namespace

    @overrides
    def _read(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as data_file:
            for row in data_file:
                row = json.loads(row)
                text1 = row["text1"]
                text2 = row["text2"]
                label = row["label"]

                instance = self.text_to_instance(sentence1=text1, sentence2=text2, label=label)
                if instance is not None:
                    yield instance

    def text_to_instance(
        self, sentence1: str, sentence2: str, label: str = None
    ) -> Instance:
        fields: Dict[str, Field] = {}

        tokens1 = self.tokenizer.tokenize(sentence1)
        tokens2 = self.tokenizer.tokenize(sentence2)
        fields['premise'] = TextField(tokens=tokens1, token_indexers=self.token_indexers)
        fields['hypothesis'] = TextField(tokens=tokens2, token_indexers=self.token_indexers)

        if label is not None:
            label_field = LabelField(label, skip_indexing=False)
            fields["label"] = label_field

        return Instance(fields)


if __name__ == '__main__':
    pass
