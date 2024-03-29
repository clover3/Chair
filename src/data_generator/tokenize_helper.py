from abc import ABC, abstractmethod
from typing import NamedTuple, List, NewType, Dict

from data_generator.tokenizer_wo_tf import get_tokenizer

g_tokenizer = None


# : Type[SubwordIndex]
SbwordIdx = NewType('SubwordIndex', int)
WordIdx = NewType('WordIdx', int)


class DocumentRep(ABC):
    @abstractmethod
    def get_word_tokens(doc, st, ed):
        pass

    @abstractmethod
    def get_subword_tokens(doc, st, ed):
        pass

    @classmethod
    @abstractmethod
    def get_duplicate(cls, doc_list: List):
        pass

    @abstractmethod
    def get_sb_len(self):
        pass


class TokenizedText(NamedTuple):
    text: str
    tokens: List[str]
    sbword_tokens: List[str]
    # idx of subword to idx of word
    sbword_mapping: List[WordIdx]

    def to_json(self) -> Dict:
        return {
            'text': self.text,
            "tokens": self.tokens,
            'sbword_tokens': self.sbword_tokens,
            'sbword_mapping': self.sbword_mapping
        }

    @classmethod
    def from_json(cls, j):
        sbword_mapping: List[WordIdx] = [WordIdx(e) for e in j['sbword_mapping']]
        return TokenizedText(
            j['text'],
            j['tokens'],
            j['sbword_tokens'],
            sbword_mapping
        )

    @classmethod
    def from_text(cls, text, tokenizer=None):
        if tokenizer is None:
            global g_tokenizer
            if g_tokenizer is None:
                g_tokenizer = get_tokenizer()
            tokenizer = g_tokenizer

        tokens = text.split()
        return cls.from_word_tokens(text, tokenizer, tokens)

    @classmethod
    def from_word_tokens(cls, text, tokenizer, tokens):
        idx_mapping = []
        subword_list = []
        for idx, token in enumerate(tokens):
            sb_tokens = tokenizer.tokenize(token)
            idx_mapping.extend([WordIdx(idx)] * len(sb_tokens))
            subword_list.extend(sb_tokens)
        return TokenizedText(text, tokens, subword_list, idx_mapping)

    def conver_sbword_indice(self, idx):
        st = self.sbword_mapping[idx]
        return st

    def get_word_tokens(doc, st: WordIdx, ed: WordIdx):
        if type(st) == WordIdx and type(ed) == WordIdx:
            if ed >= 0:
                word_tokens = doc.tokens[st: ed]
            else:
                word_tokens = doc.tokens[st:]
            return word_tokens
        else:
            raise TypeError

    def get_subword_tokens(self, st, ed):
        if type(st) == SbwordIdx and type(ed) == SbwordIdx:
            return self.sbword_tokens[st: ed]
        else:
            raise TypeError

    @classmethod
    def get_duplicate(cls, doc_list: List):
        def para_hash(doc):
            return " ".join(doc.tokens)
        hash_set = set()
        duplicates = []
        for idx, doc in enumerate(doc_list):
            hash = para_hash(doc)
            if hash in hash_set:
                duplicates.append(idx)
                continue

            hash_set.add(hash)
        return duplicates

    def get_sb_len(self):
        return len(self.sbword_tokens)


def get_duplicate(doc_list: List[TokenizedText]):
    def para_hash(doc: TokenizedText):
        return " ".join(doc.tokens)

    hash_set = set()
    duplicates = []
    for idx, doc in enumerate(doc_list):
        hash = para_hash(doc)
        if hash in hash_set:
            duplicates.append(idx)
            continue

        hash_set.add(hash)
    return duplicates

