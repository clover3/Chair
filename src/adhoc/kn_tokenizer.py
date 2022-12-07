from typing import List

from krovetzstemmer import Stemmer


class KrovetzNLTKTokenizer:
    def __init__(self):
        self.stemmer = Stemmer()

    def tokenize_stem(self, text: str) -> List[str]:
        tokens = nltk.tokenize.word_tokenize(text)
        stemmed_tokens = []
        for t in tokens:
            try:
                stemmed_tokens.append(self.stemmer.stem(t))
            except:
                pass

        return stemmed_tokens