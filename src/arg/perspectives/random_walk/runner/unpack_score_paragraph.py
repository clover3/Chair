from typing import List

from cache import load_from_pickle, save_to_pickle


def main():
    #convert("pc_dev_paras_top_100", "pc_dev_paras_top_100_list_form")
    convert("pc_train_paas_by_cid", "pc_train_paras_list_form")


def convert(input_name, output_name):
    data = load_from_pickle(input_name)

    output = {}
    for cid, para_list in data.items():
        tokens_list: List[List[str]] = [e.paragraph.tokens for e in para_list]
        output[cid] = tokens_list

    save_to_pickle(output, output_name)


if __name__ == "__main__":
    main()

