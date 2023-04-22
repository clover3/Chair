from abc import abstractmethod, ABC
from typing import Dict

import numpy as np
from transformers import AutoTokenizer


class VocaSpaceIF(ABC):
    @abstractmethod
    def dict_to_numpy(self, number: Dict[str, float]) -> np.array:
        pass

    @abstractmethod
    def numpy_to_dict(self, arr: np.array) -> np.array:
        pass

    @abstractmethod
    def get_token_id(self, token):
        pass


class BertVocaSpace(VocaSpaceIF):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.inv_vocab = {v: k for k, v in self.tokenizer.vocab.items()}
        self.voca_size = 1 + max(self.tokenizer.vocab.values())

    def dict_to_numpy(self, number: Dict[str, float]) -> np.array:
        arr = np.zeros([self.voca_size])
        for word, score in number.items():
            for subword in self.tokenizer.tokenize(word):
                token_id = self.tokenizer.vocab[subword]
                arr[token_id] += score
                break
        return arr

    def numpy_to_dict(self, arr: np.array) -> np.array:
        out_d = {}
        non_zero_indices, = arr.nonzero()
        for token_id in non_zero_indices:
            token = self.inv_vocab[token_id]
            out_d[token] = arr[token_id]
        return out_d

    def get_token_id(self, token):
        token_id = self.tokenizer.vocab[token]
        return token_id


class VocaSpace(VocaSpaceIF):
    def __init__(self, terms):
        voca: Dict[str, int] = {}
        for idx, term in enumerate(terms):
            voca[term] = idx + 1
        self.voca: Dict[str, int] = voca
        self.inv_vocab: Dict[int, str] = {v: k for k, v in voca.items()}
        self.voca_size = 1 + max(self.voca.values())

    def dict_to_numpy(self, number: Dict[str, float]) -> np.array:
        arr = np.zeros([self.voca_size])
        for word, score in number.items():
            token_id = self.voca[word]
            arr[token_id] += score
        return arr

    def numpy_to_dict(self, arr: np.array) -> np.array:
        out_d = {}
        non_zero_indices, = arr.nonzero()
        for token_id in non_zero_indices:
            token = self.inv_vocab[token_id]
            out_d[token] = arr[token_id]
        return out_d

    def get_token_id(self, token):
        token_id = self.voca[token]
        return token_id

