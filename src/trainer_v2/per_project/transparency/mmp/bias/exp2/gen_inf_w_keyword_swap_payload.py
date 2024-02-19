import sys
import xmlrpc.client
from omegaconf import OmegaConf
from iter_util import load_jsonl
from misc_lib import TELI
from trainer_v2.per_project.transparency.mmp.bias.common import load_car_maker_list, find_indices
from trainer_v2.per_project.transparency.mmp.bias.inference_w_keyword_swap import run_inference_inner2


def main():
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)

    term_list = load_car_maker_list()
    term_list_set = set(term_list)
    l_dict: list[dict] = load_jsonl(conf.source_path)
    l_dict_itr = TELI(l_dict, len(l_dict))


    def car_maker_replace(text):
        tokens = text.split()
        matching_indices = find_indices(tokens, term_list_set)
        if not matching_indices:
            print("WARNING: {} does not have matching tokens".format(text))

        for term in term_list:
            tokens_new = list(tokens)
            for i in matching_indices:
                tokens_new[i] = term
            yield " ".join(tokens_new)

    for entry in l_dict_itr:
        doc_text = entry["doc_text"]
        query_text = entry["query"]
        text_list = list(car_maker_replace(doc_text))
        tuple_itr: list[tuple[str, str]] = [(query_text, t) for t in text_list]
        for t in tuple_itr:
            print("\t".join(t))


if __name__ == "__main__":
    main()
