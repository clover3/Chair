

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
                print(t)

        return stemmed_tokens


def test_pc_tokenizer():
    t = PCTokenizer()
    text_list = ["The graduated response is a violation of the basic right to due process",
                 "ISP will not cooperate with a graduated response policy"]

    for text in text_list:
        print(text)
        print(t.tokenize_stem(text))


if __name__ == "__main__":
    test_pc_tokenizer()

