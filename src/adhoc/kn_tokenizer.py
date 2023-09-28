from typing import List
import nltk as nltk
from krovetzstemmer import Stemmer
from collections import Counter
from typing import List, Iterable, Callable, Dict, Tuple, Set
from models.classic.stopword import load_stopwords


class KrovetzNLTKTokenizer:
    def __init__(self, drop_stopwords=False):
        self.stemmer = Stemmer()
        if drop_stopwords:
            self.stopword = load_stopwords()
            # print("Drop stopwords")
        else:
            self.stopword = None

    def tokenize_stem(self, text: str) -> List[str]:
        tokens = nltk.tokenize.word_tokenize(text)
        if self.stopword is not None:
            tokens = [t for t in tokens if t.lower() not in self.stopword]
        stemmed_tokens = []
        for t in tokens:
            try:
                stemmed_tokens.append(self.stemmer.stem(t))
            except:
                pass

        return stemmed_tokens


def count_df(docs: Iterable[str]) -> Counter:
    tokenizer = KrovetzNLTKTokenizer()
    df = Counter()
    for p in docs:
        tokens = tokenizer.tokenize_stem(p)

        for term in set(tokens):
            df[term] += 1

    return df


def count_df_no_stem(docs: Iterable[str]) -> Counter:
    df = Counter()
    for p in docs:
        tokens = nltk.tokenize.word_tokenize(p)
        tokens = [t.lower() for t in tokens]

        for term in set(tokens):
            df[term] += 1

    return df
