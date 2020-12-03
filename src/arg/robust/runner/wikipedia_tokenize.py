from arg.robust.wikipedia.tokenize import jsonl_tokenize
from cache import save_to_pickle


def main():
    jsonl_path = "/mnt/nfs/work3/youngwookim/data/qck/robust_on_wiki/docs.jsonl"
    tokens_d = jsonl_tokenize(jsonl_path)
    save_to_pickle(tokens_d, "robust_on_wiki_tokens")


if __name__ == "__main__":
    main()
