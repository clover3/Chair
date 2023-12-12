import sys

from omegaconf import OmegaConf

from cache import save_list_to_jsonl
from iter_util import load_jsonl
from trainer_v2.per_project.transparency.misc_common import read_lines
from cpath import output_path
from misc_lib import path_join


def main():
    conf_path = sys.argv[1]
    config = OmegaConf.load(conf_path)

    term_list_path = path_join(output_path, "mmp", "bias", "car_exp", "car_maker_list.txt")
    term_list = [line.lower().strip() for line in open(term_list_path, "r")]
    extended_list = []

    for name in term_list:
        if name[-1] != "s":
            extended_list.append(name + "s")
    term_list += extended_list

    relevant_entries = load_jsonl(config.entry_jsonl_path)
    generic_keywords = [line.lower().strip() for line in open(config.known_generic_terms_path, "r")]

    unique_doc_text = set()

    for entry in relevant_entries:
        doc_text = entry["doc_text"].lower()
        tokens = doc_text.split()
        skip = False
        unique_mentions = set()
        for idx, token in enumerate(tokens):
            if token in term_list:
                next = tokens[idx+1]
                unique_mentions.add(token)
                if next not in generic_keywords:
                    skip = True

        if len(unique_mentions) > 1:
            skip = True

        if not skip:
            out_tokens = replace_term_to_mask(term_list, tokens)
            doc_text_masked = " ".join(out_tokens)
            unique_doc_text.add(doc_text_masked)

    for t in unique_doc_text:
        print(t)


def replace_term_to_mask(term_list, tokens):
    out_tokens = []
    for token in tokens:
        if token in term_list:
            out_tokens.append("[MASK]")
        else:
            out_tokens.append(token)
    return out_tokens


if __name__ == "__main__":
    main()