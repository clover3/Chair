from typing import List, Iterable

from contradiction.medical_claims.alamri.annotation_a import enum_true_instance
from list_lib import foreach


def load_sentences() -> Iterable[str]:
    for sent1, sent2 in enum_true_instance(1):
        yield sent1.text
        yield sent2.text


def main():
    sentences: List[str] = list(load_sentences())
    foreach(print, sentences)


if __name__ == "__main__":
    main()
