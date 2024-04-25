#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import List

from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.tokenizers.token_class import Token
import sentencepiece as spm


@Tokenizer.register("sentence_piece_bpe_tokenizer")
class SentencePieceBPETokenizer(Tokenizer):
    def __init__(self,
                 bpe_tokenizer_file: str
                 ):
        super(SentencePieceBPETokenizer, self).__init__()
        self.bpe_tokenizer_file = bpe_tokenizer_file

        bpe_tokenizer = spm.SentencePieceProcessor()
        bpe_tokenizer.Load(self.bpe_tokenizer_file)
        self.bpe_tokenizer = bpe_tokenizer

    def tokenize(self, text: str) -> List[Token]:
        tokens = self.bpe_tokenizer.EncodeAsPieces(text)
        tokens = [Token(token) for token in tokens]
        return tokens


if __name__ == "__main__":
    pass
