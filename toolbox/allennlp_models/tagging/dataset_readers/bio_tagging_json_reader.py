#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import logging
from typing import Dict, List, Optional, Sequence, Iterable, Union

from allennlp.data.dataset_readers.dataset_utils import to_bioul
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

from allennlp_models.common.ontonotes import Ontonotes, OntonotesSentence

logger = logging.getLogger(__name__)


@DatasetReader.register("bio_tagging_json")
class BioTaggingJsonReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        label_namespace: str = "labels",

    ) -> None:
        super().__init__()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.label_namespace = label_namespace

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r", encoding='utf-8') as data_file:
            for line in data_file.readlines():
                if not line:
                    continue
                items = json.loads(line)
                tokens = items["tokens"]
                tags = items.get("tags", None)

                tokens = [Token(token) for token in tokens]

                instance = self.text_to_instance(tokens=tokens, tags=tags)
                if instance is not None:
                    yield instance

    def text_to_instance(
        self, tokens: List[Token], tags: List[str] = None
    ) -> Instance:
        sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {"tokens": sequence}
        instance_fields["metadata"] = MetadataField({"words": [x.text for x in tokens]})

        if tags is not None:
            instance_fields["tags"] = SequenceLabelField(tags, sequence, self.label_namespace)

        return Instance(instance_fields)


if __name__ == '__main__':
    pass
