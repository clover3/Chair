

# Include tokenization, normalization stemming
from typing import List

import nltk
from krovetzstemmer import Stemmer

from arg.perspectives.kn_tokenizer import KrovetzNLTKTokenizer
from models.classic.stemming import StemmedToken


class PCTokenizerEx:
    def __init__(self, on_unicode_error="skip"):
        self.stemmer = Stemmer()
        self.on_unicode_error = on_unicode_error

    def tokenize_stem(self, text: str) -> List[StemmedToken]:
        tokens = nltk.tokenize.word_tokenize(text)
        stemmed_tokens = []
        for t in tokens:
            try:
                t_: StemmedToken = StemmedToken(t, self.stemmer.stem(t))
                stemmed_tokens.append(t_)
            except UnicodeDecodeError:
                if self.on_unicode_error == "skip":
                    pass
                elif self.on_unicode_error == "keep":
                    stemmed_tokens.append(t)
                elif self.on_unicode_error == "raise":
                    raise
                else:
                    assert False

        return stemmed_tokens


obj_pc_tokenizer_ex = None


def pc_tokenize_ex(text) -> List[StemmedToken]:
    global obj_pc_tokenizer_ex
    if obj_pc_tokenizer_ex is None:
        obj_pc_tokenizer_ex = PCTokenizerEx()

    return obj_pc_tokenizer_ex.tokenize_stem(text)



def test_pc_tokenizer():
    t = KrovetzNLTKTokenizer()
    text_list = ["The graduated response is a violation of the basic right to due process",
                 "ISP will not cooperate with a graduated response policy"]

    for text in text_list:
        print(text)
        print(t.tokenize_stem(text))


if __name__ == "__main__":
    test_pc_tokenizer()

