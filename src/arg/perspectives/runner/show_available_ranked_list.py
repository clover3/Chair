from typing import List

from arg.perspectives.basic_analysis import load_data_point, PerspectiveCandidate
from arg.perspectives.ranked_list_interface import StaticRankedListInterface
from galagos.query_runs_ids import Q_CONFIG_ID_BM25_10000


def show():
    ci = StaticRankedListInterface(Q_CONFIG_ID_BM25_10000)
    all_data_points: List[PerspectiveCandidate] = load_data_point("dev")

    pre_cid = -1
    n_found = 0
    n_not_found = 0
    l = []
    for x in all_data_points:
        if x.cid != pre_cid:
            l.append((pre_cid, n_found, n_not_found))
            n_found = 0
            n_not_found = 0
            pre_cid = x.cid
        try:
            ci.fetch(x.cid, x.pid)
            n_found += 1
        except KeyError:
            n_not_found += 1

    l.sort(key=lambda x:x[0])
    print("{} datapoints".format(len(all_data_points)))
    print("{} claims".format(len(l)))
    for e in l:
        print(e)

if __name__ == "__main__":
    show()