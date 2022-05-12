from typing import List

import torch

from ptorch.ists.chunk_embedding import get_bert_sentence_embedding, get_bert_chunks_embedding3


class BertChunkEncoderForInference:
    def __init__(self, bert_tokenizer, bert_model):
        self.bert_tokenizer = bert_tokenizer
        self.bert_model = bert_model

    def encode(self, sentence_as_chunk: List[str], target_len=None) -> torch.Tensor:
        if target_len is not None:
            sentence_as_chunk = sentence_as_chunk[:target_len]
        sentence = " ".join(sentence_as_chunk)
        chunks_as_st_ed = []
        cur_idx = 0
        for chunk_s in sentence_as_chunk:
            tokens = chunk_s.split()
            chunk_len = len(tokens)
            end_idx = cur_idx + chunk_len - 1
            chunks_as_st_ed.append((cur_idx, end_idx))
            cur_idx += chunk_len
        tokenized_text, sentence_embedding = get_bert_sentence_embedding(sentence, self.bert_tokenizer, self.bert_model)
        try:
            matrix = get_bert_chunks_embedding3(sentence_embedding, chunks_as_st_ed)
        except IndexError:
            for chunk_s in sentence_as_chunk:
                tokens = self.bert_tokenizer.wordpiece_tokenizer.tokenize(chunk_s)
                chunk_len = len(tokens)
                end_idx = cur_idx + chunk_len - 1
                print(cur_idx, end_idx, tokens, chunk_len)
                chunks_as_st_ed.append((cur_idx, end_idx))
                cur_idx += chunk_len

            raise

        if target_len is not None:
            chunk_dim = matrix.shape[1]
            pad_len = target_len - len(matrix)
            pad_mat = torch.zeros([pad_len, chunk_dim])
            matrix = torch.cat([matrix, pad_mat])
        return matrix
