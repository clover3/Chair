import spacy

from data_generator2.segmented_enc.sent_split_by_spacy import split_spacy_tokens
from dataset_specific.mnli.mnli_reader import MNLIReader


def main():
    nlp = spacy.load("en_core_web_sm")
    reader = MNLIReader()
    iter = reader.load_split("train")
    for idx, e in enumerate(iter):
        print()
        print(idx, e)
        print(e)
        tokens = nlp(e.hypothesis)
        split_spacy_tokens(tokens)

    return NotImplemented


if __name__ == "__main__":
    main()
