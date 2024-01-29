import sys
from collections import Counter

from cpath import output_path
from list_lib import right
from misc_lib import path_join, get_second
from tab_print import print_table
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.mmp.bias.common import load_car_maker_list




def analyze_inner(passage_dict, query_dict, score_log, term_list):
    left_based_count = Counter()
    for term_a in term_list:
        for term_b in term_list:
            if term_a != term_b and term_a not in term_b and term_b not in term_a:
                counter = count_wins(passage_dict, score_log, term_list, term_a, term_b)
                print(term_a, term_b, counter[True], counter[False])
                left_based_count[term_a, True] += counter[True]
                left_based_count[term_a, False] += counter[False]

    table = []
    for term_a in term_list:
        win = left_based_count[term_a, True]
        loss = left_based_count[term_a, False]
        row = term_a, win / (win+loss)
        table.append(row)

    table.sort(key=get_second, reverse=True)
    print_table(table)


def count_wins(passage_dict, score_log, term_list, target_term_a, target_term_b):
    seen_doc = set()
    counter = Counter()
    for row in score_log:
        doc_id = row[1]
        if doc_id in seen_doc:
            continue

        seen_doc.add(doc_id)
        q_id = row[0]
        scores = list(map(float, row[2:]))
        doc_text = passage_dict[doc_id]
        doc_tokens = doc_text.split()
        originally_matched_terms = [term for idx, term in enumerate(term_list) if term in doc_tokens]
        if target_term_a in originally_matched_terms or target_term_b in originally_matched_terms:
            continue
        if max(scores) < 0:
            continue
        # print("Relevant")
        entry_list = []
        for idx, s in enumerate(scores):
            entry = [term_list[idx], scores[idx]]
            entry_list.append(entry)
        entry_list.sort(key=get_second, reverse=True)
        # if high indices has something other than originally matched terms, it is suspicious
        entity_to_score = dict(entry_list)
        win = entity_to_score[target_term_a] > entity_to_score[target_term_b]
        # print(f"query {q_id}: {query_dict[q_id]}")
        # print(f"passage {doc_id}: {passage_dict[doc_id]}")
        # print("ford > honda", )
        # print("Original: " + ", ".join(right(originally_matched_terms)))

        # if win and target_term_a in doc_text.lower():
        #     pass
        # elif not win and target_term_b in doc_text.lower():
        #     pass
        # else:
        counter[win] += 1

        # print("High scored: ")
        # s = ", ".join(["{0} ({1:.2f})".format(term, score) for term, score in entry_list])
        # print(s)
        # print()
    return counter


def main():
    file_path = sys.argv[1]
    score_log = list(tsv_iter(file_path))
    term_list = load_car_maker_list()
    term_list_set = set(term_list)
    passage_dict = dict(tsv_iter(path_join(output_path, "mmp", "bias", "car_exp", "car_maker_texts.tsv")))
    query_dict = dict(tsv_iter(path_join(output_path, "mmp", "bias", "car_exp", "car_queries_no_maker.tsv")))
    analyze_inner(passage_dict, query_dict, score_log, term_list)


if __name__ == "__main__":
    main()
