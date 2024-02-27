import json
from collections import Counter
from typing import NamedTuple, List, Tuple, Dict

from list_lib import lmap


class TokenizedText(NamedTuple):
    text: str
    sp_tokens: List[str]
    sb_tokens: List[List[str]]

    sbword_mapping: List[int] # idx of subword to idx of word
    sp_sb_mapping: List[List[int]]

    @classmethod
    def from_text(cls, tokenizer, text):
        sp_tokenize = tokenizer.basic_tokenizer.tokenize
        sb_tokenize = tokenizer.wordpiece_tokenizer.tokenize
        sp_tokens: list[str] = sp_tokenize(text)
        sb_tokens: List[List[str]] = lmap(sb_tokenize, sp_tokens)

        sbword_mapping = []
        sp_sb_mapping = []
        for sp_idx, sb_tokens_per_sp_token in enumerate(sb_tokens):
            mapping_per_sp = []
            for _ in sb_tokens_per_sp_token:
                n_sb_before = len(sbword_mapping)
                mapping_per_sp.append(n_sb_before)
                sbword_mapping.append(sp_idx)
            sp_sb_mapping.append(mapping_per_sp)
        return TokenizedText(text, sp_tokens, sb_tokens, sbword_mapping, sp_sb_mapping)

    def get_sb_range(self, sp_idx) -> Tuple[int, int]:
        st = self.sp_sb_mapping[sp_idx][0]
        if sp_idx + 1 < len(self.sp_sb_mapping):
            ed = self.sp_sb_mapping[sp_idx + 1][0]
        else:
            ed = self.get_sb_len()
        return st, ed

    def get_sb_list_form_sp_indices(self, sp_indices) -> List[str]:
        output = []
        for i in sp_indices:
            output.extend(self.sb_tokens[i])
        return output

    def get_sb_len(self):
        return sum(map(len, self.sb_tokens))

    def __str__(self) -> str:
        return (
            f"TokenizedText(\n"
            f"\ttext: {self.text},\n"
            f"\tsp_tokens: {self.sp_tokens},\n"
            f"\tsb_tokens: {self.sb_tokens},\n"
            f"\tsbword_mapping: {self.sbword_mapping},\n"
            f"\tsp_sb_mapping: {self.sp_sb_mapping}\n"
            f")"
        )

    def get_sp_to_sb_map(self) -> dict[str, list[str]]:
        out_d = {}
        for sp, sb in zip(self.sp_tokens, self.sb_tokens):
            out_d[sp] = sb
        return out_d


def translate_sp_idx_to_sb_idx(sb_tokens: List[List[str]], st: int, ed: int):
    sbword_mapping = []
    sp_sb_mapping = []
    for sp_idx, sb_tokens_per_sp_token in enumerate(sb_tokens):
        mapping_per_sp = []
        if not sb_tokens_per_sp_token:
            print(sb_tokens)
            raise ValueError()
        for _ in sb_tokens_per_sp_token:
            n_sb_before = len(sbword_mapping)
            mapping_per_sp.append(n_sb_before)
            sbword_mapping.append(sp_idx)
        sp_sb_mapping.append(mapping_per_sp)

    sb_st = sp_sb_mapping[st][0]
    try:
        sb_ed = sp_sb_mapping[ed][0] if ed < len(sb_tokens) else len(sbword_mapping)
    except IndexError:
        print("st, ed", st, ed)
        print("len(sb_tokens)", sb_tokens)
        print("sp_sb_mapping", sp_sb_mapping)
        raise
    return sb_st, sb_ed


def merge_subword_scores(scores, text: TokenizedText, merge_method):
    output = []
    for indices in text.sp_sb_mapping:
        new_score = merge_method([scores[i] for i in indices])
        output.append(new_score)
    return output


def is_valid_indices(text: TokenizedText, sp_indices):
    for i in sp_indices:
        if 0 <= i < len(text.sp_tokens):
            pass
        else:
            return False
    return True


def enum_neighbor(idx):
    yield [idx - 1, idx]
    yield [idx, idx + 1]


def get_term_rep(rep: TokenizedText, indices):
    return " ".join([rep.sp_tokens[i] for i in indices])


class TrialLogger:
    def __init__(self, save_path):
        self.f = open(save_path, "a")

    def log_rep(self, q_rep: TokenizedText, d_rep: TokenizedText):
        j = {'query': q_rep.text, "document": d_rep.text}
        self.f.write(json.dumps(j) + "\n")

    def log_score(self, q_indices, d_indices, score):
        j = {'q_indices': q_indices, "d_indices": d_indices, "score": score}
        self.f.write(json.dumps(j) + "\n")


class TextRep:
    def __init__(self, tokenized_text: TokenizedText):
        self.tokenized_text = tokenized_text
        self.counter = Counter(self.tokenized_text.sp_tokens)

        indices = {t: list() for t in self.counter}
        for idx, sp_token in enumerate(self.tokenized_text.sp_tokens):
            indices[sp_token].append(idx)
        self.indices: Dict[str, List[int]] = indices
        self.keys = list(self.indices.keys())
        self.keys.sort()

    def get_bow(self):
        for term, cnt in self.counter.items():
            yield term, cnt, self.indices[term]

    def get_terms(self):
        return self.keys

    @classmethod
    def from_text(cls, tokenizer, text):
        tt = TokenizedText.from_text(tokenizer, text)
        return TextRep(tt)

    def get_sp_size(self):
        return len(self.tokenized_text.sp_tokens)