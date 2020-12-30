

# Include tokenization, normalization stemming
from typing import List

import nltk
from krovetzstemmer import Stemmer


class PCTokenizer:
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


class StemmedToken:
    def __init__(self, raw_token, stemmed_token):
        self.raw_token = raw_token
        self.stemmed_token = stemmed_token

    def __str__(self):
        return self.stemmed_token

    def get_raw_token(self):
        return self.raw_token

    def get_stemmed_token(self):
        return self.stemmed_token


class PCTokenizerEx:
    def __init__(self):
        self.stemmer = Stemmer()

    def tokenize_stem(self, text: str) -> List[StemmedToken]:
        tokens = nltk.tokenize.word_tokenize(text)
        stemmed_tokens = []
        for t in tokens:
            try:
                t_: StemmedToken = StemmedToken(t, self.stemmer.stem(t))
                stemmed_tokens.append(t_)
            except:
                pass

        return stemmed_tokens


obj_pc_tokenizer_ex = None


def pc_tokenize_ex(text) -> List[StemmedToken]:
    global obj_pc_tokenizer_ex
    if obj_pc_tokenizer_ex is None:
        obj_pc_tokenizer_ex = PCTokenizerEx()

    return obj_pc_tokenizer_ex.tokenize_stem(text)



def test_pc_tokenizer():
    t = PCTokenizer()
    text_list = ["The graduated response is a violation of the basic right to due process",
                 "ISP will not cooperate with a graduated response policy"]

    for text in text_list:
        print(text)
        print(t.tokenize_stem(text))


if __name__ == "__main__":
    test_pc_tokenizer()

