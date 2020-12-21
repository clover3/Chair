import re
from typing import List, Iterable, Dict, Tuple, NamedTuple

import nltk

from list_lib import lmap
from models.classic.stopword import load_stopwords_for_query


class Tokens(NamedTuple):
    tokens: List[str]
    text: str


class NLTKTokenizer:
    def __init__(self):
        pass

    def tokenize(self, text) -> Tokens:
        tokens = nltk.tokenize.word_tokenize(text.lower())
        return Tokens(tokens, text)


class PrenormalizeTokenizer:
    def __init__(self, normalize_dict: Dict[str, str]):

        self.normalize_entries = []
        for source_text, normal_text in normalize_dict.items():
            reg_str = r"([\s^])" + source_text + "([\\s,\\.])"
            print(reg_str)
            rec = re.compile(reg_str)
            self.normalize_entries.append((rec, normal_text))

        pass

    def tokenize(self, text) -> Tokens:
        cur_text = text
        for rec, normal_text in self.normalize_entries:
            after = "\\1" + normal_text+"\\2"
            cur_text = rec.sub(after, cur_text)

        tokens = nltk.tokenize.word_tokenize(cur_text.lower())
        return Tokens(tokens, text)


stopwords = None


def filter_stopwords(tokens: Iterable[str]) -> List[str]:
    global stopwords
    if stopwords is None:
        stopwords = load_stopwords_for_query()
    return list([t for t in tokens if t not in stopwords])


def compare(tokens_answer: List[str], tokens_target: List[str]):
    uncovered = [t for t in tokens_answer if t not in tokens_target]
    extra = [t for t in tokens_target if t not in tokens_answer]
    uncovered = filter_stopwords(uncovered)
    extra = filter_stopwords(extra)
    return uncovered, extra


def compare_bidirection(tokens_a: List[str], tokens_b: List[str]):
    mismatch_a = [t for t in tokens_a if t not in tokens_b]
    mismatch_b = [t for t in tokens_b if t not in tokens_a]
    mismatch_a = filter_stopwords(mismatch_a)
    mismatch_b = filter_stopwords(mismatch_b)
    return mismatch_a, mismatch_b


d_full = {
        'L Arginine': 'l-arginine',
        'L-Arg': 'l-arginine',
        'pre-eclamptic': 'pre-eclampsia',
        'preeclampsia': 'pre-eclampsia',
        'pre-eclampsia': 'pre-eclampsia pregnant women',
        'gestational': 'pregnant',
        'pregnancy': 'pregnant',
        'pregnant': 'pregnant women'
    }


d_stemming = {
        'L Arginine': 'l-arginine',
        'L-Arg': 'l-arginine',
        'pre-eclamptic': 'pre-eclampsia',
        'preeclampsia': 'pre-eclampsia',
        'pre-eclampsia': 'pre-eclampsia',
        'pregnancy': 'pregnant',
    }


def solve(question: str, claims: List[str]):
    #  tokenize
    #  compare and check which terms are extra

    tokenizer = PrenormalizeTokenizer(d_full)
    question_tokens: Tokens = tokenizer.tokenize(question)
    claims_tokens: List[Tokens] = lmap(tokenizer.tokenize, claims)

    for c_tokens in claims_tokens:
        uncovered, extra = compare(question_tokens.tokens, c_tokens.tokens)
        print()
        print("Claim:", c_tokens.text)
        print("uncovered:", uncovered)
        print("extra:", extra)


def pairwise(claim_pairs: List[Tuple[str,str]]):
    tokenizer = PrenormalizeTokenizer(d_stemming)
    for c1, c2 in claim_pairs:
        c1_tokens = tokenizer.tokenize(c1)
        c2_tokens = tokenizer.tokenize(c2)
        mismatch_1, mismatch_2 = compare_bidirection(c1_tokens.tokens, c2_tokens.tokens)
        print()
        print("Claim A:", c1_tokens.text)
        print("mismatch:", mismatch_1)
        print("Claim B:", c2_tokens.text)
        print("mismatch:", mismatch_2)

