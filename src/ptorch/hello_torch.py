from sentence_transformers import SentenceTransformer

from misc_lib import tprint


def main():
    sentences = ["This is an example sentence", "Each sentence is converted"]
    sentences100 = sentences * 100
    model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

    tprint("Begin")
    embeddings = model.encode(sentences100)
    tprint("End")


if __name__ == "__main__":
    main()