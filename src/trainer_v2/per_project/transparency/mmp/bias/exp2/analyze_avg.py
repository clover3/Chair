import sys
from collections import Counter, defaultdict

from cpath import output_path
from list_lib import right
from misc_lib import path_join, get_second, average
from tab_print import print_table
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.mmp.bias.common import load_car_maker_list




def analyze_inner(score_log, term_list):
    seen_doc = set()
    entity_scores = defaultdict(list)
    for row in score_log:
        doc_id = row[1]
        if doc_id in seen_doc:
            continue

        seen_doc.add(doc_id)
        scores = list(map(float, row[2:]))
        if max(scores) < 0:
            continue
        for idx, s in enumerate(scores):
            entity_scores[term_list[idx]].append(scores[idx])

    avg_scores = [(k, average(v)) for k, v in entity_scores.items()]
    avg_scores.sort(key=get_second, reverse=True)
    print_table(avg_scores)



def main():
    file_path = sys.argv[1]
    score_log = list(tsv_iter(file_path))
    term_list = load_car_maker_list()
    analyze_inner(score_log, term_list)


if __name__ == "__main__":
    main()
