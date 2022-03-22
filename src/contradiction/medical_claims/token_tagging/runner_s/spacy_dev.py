from typing import Iterable

import spacy

from contradiction.ie_align.srl.spacy_segmentation import spacy_segment
from contradiction.medical_claims.token_tagging.problem_loader import iter_unique_text


def main():
    split = "dev"
    nlp = spacy.load("en_core_web_sm")
    text_list: Iterable[str] = iter_unique_text(split)
    for text in text_list:
        print(text)
        doc = nlp(text)
        spacy_segment(doc)


if __name__ == "__main__":
    main()