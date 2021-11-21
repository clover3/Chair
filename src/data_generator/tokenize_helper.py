from typing import NamedTuple, List, NewType

from data_generator.tokenizer_wo_tf import get_tokenizer

g_tokenizer = None


class TokenizedText(NamedTuple):
    text: str
    tokens: List[str]
    sbword_tokens: List[str]
    # idx of subword to idx of word
    sbword_mapping: List[int]

    @classmethod
    def from_text(cls, text, tokenizer=None):
        if tokenizer is None:
            global g_tokenizer
            if g_tokenizer is None:
                g_tokenizer = get_tokenizer()
            tokenizer = g_tokenizer

        tokens = text.split()
        idx_mapping = []
        subword_list = []
        for idx, token in enumerate(tokens):
            sb_tokens = tokenizer.tokenize(token)
            idx_mapping.extend([idx] * len(sb_tokens))
            subword_list.extend(sb_tokens)

        return TokenizedText(text, tokens, subword_list, idx_mapping)

    def conver_sbword_indice(self, idx):
        st = self.sbword_mapping[idx]
        return st


# : Type[SubwordIndex]
SbwordIdx = NewType('SubwordIndex', int)
WordIdx = NewType('WordIdx', int)


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

