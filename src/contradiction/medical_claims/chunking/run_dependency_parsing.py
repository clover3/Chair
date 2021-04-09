from typing import List

import spacy

from cache import save_to_pickle
from contradiction.medical_claims.load_corpus import Claim, load_all_claims


def parse_text(nlp, raw_text: str):
    # TODO : apply parsing for texts
    # TODO : re-arrange parsing to serializable format
    return nlp(raw_text)


def main():
    # load sentences with id
    nlp = spacy.load("en_core_web_sm")
    claims: List[Claim] = load_all_claims()

    parsing_dictionary = {}
    for t in claims:
        parsed = parse_text(nlp, t.text)
        print(parsed)
        parsing_dictionary[t.pmid] = parsed

    # TODO : save parsing info to json
    save_to_pickle(parsing_dictionary, "medical_claim_dependency")


if __name__ == "__main__":
    main()

