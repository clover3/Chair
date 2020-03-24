from typing import List

import nltk

from galagos.parse import clean_query


def clean_tokenize_str_to_tokens(raw_str: str) -> List[str]:
    terms = clean_query(nltk.word_tokenize(raw_str))
    terms = [t.lower() for t in terms]
    return terms

