import sys
from collections import Counter

from cache import save_list_to_jsonl
from cpath import output_path
from iter_util import load_jsonl
from misc_lib import path_join
from trainer_v2.per_project.transparency.misc_common import read_lines
from trainer_v2.per_project.transparency.mmp.bias.exp2.filter_generic_metions import replace_term_to_mask
from trainer_v2.per_project.transparency.mmp.bias.exp2.print_brand_n_gram import load_car_brand_ex


def load_labels(file_path):
    label_list = []
    for line in open(file_path, "r"):
        if "yes" in line.lower():
            label_list.append(1)
        elif "no" in line.lower():
            label_list.append(0)
        else:
            print("Cannot parse: ", line)
    return label_list


def main():
    label_path = path_join(output_path, "mmp", "bias", "car_exp", "generic_mention_text_gpt4_label.txt")
    labels = load_labels(label_path)
    text_path = path_join(output_path, "mmp", "bias", "car_exp", "generic_mention_text.txt")
    lines = read_lines(text_path)
    print(Counter(labels))
    assert len(lines) == len(labels)
    rel_entries_path = path_join(output_path, "mmp", "bias", "car_exp", "relevant_filtered.jsonl")
    relevant_entries = load_jsonl(rel_entries_path)

    label_d = dict(zip(lines, labels))
    term_list_path = path_join(output_path, "mmp", "bias", "car_exp", "car_maker_list.txt")
    term_list = load_car_brand_ex(term_list_path)

    items = []
    generic_text = []
    for entry in relevant_entries:
        doc_text = entry["doc_text"].lower()
        out_tokens = replace_term_to_mask(term_list, doc_text.split())
        doc_text_masked = " ".join(out_tokens)

        try:
            if not label_d[doc_text_masked]:
                text = entry["doc_text"]
                if text not in generic_text:
                    generic_text.append(text)

                items.append(entry)
        except KeyError:
            pass

    save_path = path_join(output_path, "mmp", "bias", "car_exp", "generic_mention_selected.jsonl")
    save_list_to_jsonl(items, save_path)

    save_path = path_join(output_path, "mmp", "bias", "car_exp", "generic_text.txt")
    open(save_path, "w").writelines([t+"\n" for t in generic_text])


if __name__ == "__main__":
    main()
