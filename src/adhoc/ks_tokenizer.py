from krovetzstemmer import Stemmer

from models.classic.stopword import load_stopwords
from typing import List, Iterable, Callable, Dict, Tuple, Set


class KrovetzSpaceTokenizer:
    def __init__(self, drop_stopwords=False):
        self.stemmer = Stemmer()
        if drop_stopwords:
            self.stopword = load_stopwords()
            # print("Drop stopwords")
        else:
            self.stopword = None

    def tokenize_stem(self, text: str) -> List[str]:
        tokens = text.split()
        if self.stopword is not None:
            tokens = [t for t in tokens if t.lower() not in self.stopword]
        stemmed_tokens = []
        for t in tokens:
            try:
                stemmed_tokens.append(self.stemmer.stem(t))
            except:
                pass

        return stemmed_tokens
