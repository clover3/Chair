from typing import List

import nltk

from galagos.parse import clean_query
from models.classic.stopword import load_stopwords_for_query


def clean_tokenize_str_to_tokens(raw_str: str) -> List[str]:
    terms = clean_query(nltk.word_tokenize(raw_str))
    terms = [t.lower() for t in terms]
    return terms


def clean_text_for_query(raw_str: str) -> str:
    terms = clean_query(nltk.word_tokenize(raw_str))
    terms = " ".join([t.lower() for t in terms])
    return terms


def drop_words(q_terms, words):
    q_terms = list([t for t in q_terms if t not in words])
    return q_terms


class TokenizerForGalago:
    def __init__(self, drop_stopwords=True):
        self.drop_stopwords = drop_stopwords
        self.stopword = load_stopwords_for_query()

    def tokenize(self, text):
        tokens = nltk.tokenize.word_tokenize(text.lower())
        if self.drop_stopwords:
            tokens = drop_words(tokens, self.stopword)

        terms = clean_query(tokens)
        out_terms = []
        for t in terms:
            t = t.replace("'", "")
            if t:
                out_terms.append(t)
        return out_terms
