from collections import defaultdict, Counter

from cpath import output_path
from list_lib import right
from misc_lib import path_join, get_first
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.mmp.bias.common import load_car_maker_list
from trainer_v2.per_project.transparency.mmp.bias.common import load_from_bias_dir


def get_occurrence_idx(tokens: list[str], maker_keywords: list[str]):
    indices = []
    for idx, t in enumerate(tokens):
        for k in maker_keywords:
            if k in t:
                indices.append(idx)
                break
    return indices


def load_car_maker_related_keywords():
    term_list = set(load_car_maker_list())
    term_list.update(load_from_bias_dir("car_maker_ex.txt"))
    term_list.update(load_from_bias_dir("not_generic_car_keywords3.txt"))
    return term_list


def main():
    car_maker_list = load_car_maker_list()
    passage_path = path_join(output_path, "mmp", "bias", "car_maker_texts.tsv")
    doc_id_passage_list = list(tsv_iter(passage_path))
    known_car_model_list = load_car_maker_related_keywords()
    known_generic_car_keyword = set(load_from_bias_dir("generic_car_keywords.txt"))

    doc_text_list = right(doc_id_passage_list)
    second_terms = set()
    bigram_count = Counter()
    for doc_text in doc_text_list:
        tokens = doc_text.lower().split()
        indices = get_occurrence_idx(tokens, car_maker_list)
        bigram_list = []
        for i in indices:
            try:
                bigram = tokens[i], tokens[i+1]
                maker_name = tokens[i]
                following_token = tokens[i+1]
                if following_token in known_car_model_list:
                    pass
                elif following_token in known_generic_car_keyword:
                    pass
                elif following_token in second_terms:
                    pass
                else:
                    bigram_count[maker_name, following_token] += 1
                    #
                    # second_terms.add(tokens[i+1])
                    # if i+2 < len(tokens):
                    #     out_msg = " ".join(tokens[i:i+3])
                    # else:
                    #     out_msg = " ".join(bigram)
                    # print(out_msg)
            except IndexError:
                pass

    keys = list(bigram_count.keys())
    keys.sort(key=get_first)

    for car_maker, second_term in keys:
        print(car_maker, second_term, bigram_count[car_maker, second_term])



if __name__ == "__main__":
    main()
