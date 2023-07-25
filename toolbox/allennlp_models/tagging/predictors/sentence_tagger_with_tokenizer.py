#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import List, Dict
from copy import deepcopy

from overrides import overrides
import numpy

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance

from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.tokenizers.tokenizer import Tokenizer

from allennlp.models import Model
from allennlp.predictors.predictor import Predictor


@Predictor.register('sentence_tagger_with_tokenizer')
class SentenceTaggerWithTokenizerPredictor(Predictor):
    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader,
                 tokenizer: Tokenizer
                 ) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = tokenizer

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        tokens = self._tokenizer.tokenize(sentence)
        return self._dataset_reader.text_to_instance(tokens)

    @overrides
    def predictions_to_labeled_instances(self,
                                         instance: Instance,
                                         outputs: Dict[str, numpy.ndarray]) -> List[Instance]:
        """
        This function currently only handles BIO tags.
        https://zhuanlan.zhihu.com/p/147537898
        """
        predicted_tags = outputs['tags']
        predicted_spans = []

        i = 0
        while i < len(predicted_tags):
            tag = predicted_tags[i]
            if tag[0] == 'B':
                begin_idx = i
                while tag[0] != 'O':
                    i += 1
                    tag = predicted_tags[i]
                end_idx = i - 1

                current_tags = [t if begin_idx <= idx <= end_idx else 'O'
                                for idx, t in enumerate(predicted_tags)]
                predicted_spans.append(current_tags)
            i += 1

        # Creates a new instance for each contiguous tag
        instances = []
        for labels in predicted_spans:
            new_instance = deepcopy(instance)
            text_field: TextField = instance['tokens']
            new_instance.add_field('tags', SequenceLabelField(labels, text_field), self._model.vocab)
            instances.append(new_instance)
        instances.reverse()

        return instances

    def bio_decode(self, outputs: Dict[str, numpy.ndarray]):
        """
        This function currently only handles BIO tags.
        https://zhuanlan.zhihu.com/p/147537898
        """
        words = outputs['words']
        predicted_tags = outputs['tags']
        predicted_spans = []

        i = 0
        while i < len(predicted_tags):
            tag = predicted_tags[i]
            if tag[0] == 'B':
                entity_name = tag[2:]
                begin_idx = i

                i += 1
                tag = predicted_tags[i]

                while tag[0] == 'I':
                    i += 1
                    tag = predicted_tags[i]
                end_idx = i

                sub_words = words[begin_idx: end_idx]
                sub_words = [sub_word[2:] if sub_word.startswith('##') else sub_word for sub_word in sub_words]
                sub_words = ''.join(sub_words)

                predicted_spans.append((sub_words, entity_name))

                i -= 1
            i += 1

        return predicted_spans
