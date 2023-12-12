import sys
from cpath import output_path
from misc_lib import path_join

from cache import save_list_to_jsonl
from iter_util import load_jsonl


def main():

    relevant_entries = load_jsonl(sys.argv[1])
    term_list_path = path_join(output_path, "mmp", "bias", "car_exp", "car_maker_list.txt")
    save_path = sys.argv[2]
    term_list = load_car_brand_ex(term_list_path)

    items = []
    for entry in relevant_entries:
        doc_text = entry["doc_text"].lower()
        tokens = doc_text.split()
        for idx, token in enumerate(tokens):
            if token in term_list:
                ed = min(idx + 2, len(tokens))
                n_gram = tokens[idx: ed]
                if len(n_gram) > 1:
                    items.append(" ".join(n_gram))

    items = list(set(items))
    items.sort()

    f = open(save_path, "w")
    for ngram in items:
        f.write(ngram + "\n")


def load_car_brand_ex(term_list_path):
    term_list = [line.lower().strip() for line in open(term_list_path, "r")]
    extended_list = []
    for name in term_list:
        if name[-1] != "s":
            extended_list.append(name + "s")
    term_list += extended_list
    return term_list


if __name__ == "__main__":
    main()