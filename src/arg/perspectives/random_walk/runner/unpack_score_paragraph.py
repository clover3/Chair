from typing import List

from cache import load_from_pickle, save_to_pickle


def main():
    data = load_from_pickle("pc_dev_paras_top_100")

    output = {}
    for cid, para_list in data.items():
        tokens_list: List[List[str]] = [e.paragraph.tokens for e in para_list]
        output[cid] = tokens_list

    save_to_pickle(output, "pc_dev_paras_top_100_list_form")


if __name__ == "__main__":
    main()

