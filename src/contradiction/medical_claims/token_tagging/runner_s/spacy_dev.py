from typing import Iterable

import spacy

from contradiction.medical_claims.token_tagging.problem_loader import iter_unique_text
from trainer_v2.epr.spacy_segmentation import spacy_segment


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