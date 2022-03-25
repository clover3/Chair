from sentence_transformers import SentenceTransformer

from cache import save_to_pickle


def get_hello_world():
    return "hello world"


def main():
    model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')
    emb = model.encode(get_hello_world())
    save_to_pickle(emb, "hello_world_sent_transformer")


if __name__ == "__main__":
    main()
