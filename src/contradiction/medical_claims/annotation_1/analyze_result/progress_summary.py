import re
from collections import Counter
from typing import List, Tuple

from cache import load_from_pickle
from contradiction.medical_claims.annotation_1.load_data import get_pair_dict
from contradiction.medical_claims.annotation_1.mturk_scheme import load_all_annotation
from contradiction.medical_claims.annotation_1.worker_id_info import get_worker_list_to_reject
from contradiction.medical_claims.annotation_1.worker_id_info import trusted_worker
from list_lib import right
from mturk.parse_util import HitResult
from tab_print import print_table


def get_all_jobs() -> List[Tuple[int, int]]:
    return list(get_pair_dict().keys())

ecc_url_reg = re.compile(r"https://ecc.neocities.org/(\d+/\d+).html")




def get_done_jobs():
    all_hits: List[HitResult] = load_all_annotation()
    n_total = len(all_hits)
    hits_by_trusted_worker = [h for h in all_hits if h.worker_id in trusted_worker]
    unauthorized_worker = get_worker_list_to_reject()
    hits_by_unauthorized_worker = [h for h in all_hits if h.worker_id in unauthorized_worker]
    rows = [
        ['total accepted', n_total],
        ['hits by trusted workers', len(hits_by_trusted_worker)],
        ['hits_by_unauthorized_worker', len(hits_by_unauthorized_worker)]
    ]
    print_table(rows)
    counter = Counter()
    for h in hits_by_trusted_worker:
        match = ecc_url_reg.search(h.inputs["url"])
        input_id = match.group(1)
        input_id_tuple = input_id_str_parse(input_id)
        counter[input_id_tuple] += 1
    return counter


def input_id_str_parse(input_id):
    group_no_s, idx_s = input_id.split("/")
    input_id_tuple = int(group_no_s), int(idx_s)
    return input_id_tuple


def get_available_jobs():
    save_name = "hits_all_available"
    hits = load_from_pickle(save_name)
    counter = Counter()
    for hit in hits:
        match = ecc_url_reg.search(hit['Question'])
        if match is not None:
            input_id = match.group(1)
            input_id_tuple = input_id_str_parse(input_id)
            if hit['HITStatus'] == 'Assignable':
                counter[input_id_tuple] += hit['NumberOfAssignmentsAvailable']
    return counter


def main():
    # Read csv files
    #   Among the accepted hits.
    #   Count ones from reliable annotators.
    all_jobs = get_all_jobs()

    done_jobs_count = get_done_jobs()
    # for key in all_jobs:
    #     print(key, done_jobs_count[key])

    print("--- available ---")
    available_jobs_count = get_available_jobs()
    # for key in all_jobs:
    #     if available_jobs_count[key]:
    #         print(key, available_jobs_count[key])

    print("Total available", sum(available_jobs_count.values()))

    n_goal = 3
    to_release = []
    for key in all_jobs:
        n_to_release = n_goal - available_jobs_count[key] - done_jobs_count[key]
        to_release.append((key, n_to_release))

    # print("--- to release --")
    # print_table(to_release)
    print("Total of {} should be released".format(sum(right(to_release))))


if __name__ == "__main__":
    main()