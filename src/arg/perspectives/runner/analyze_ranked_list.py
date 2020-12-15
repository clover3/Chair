import sys
from typing import List, Dict

from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import get_claim_perspective_id_dict, get_all_claims, claims_to_dict
from trec.trec_parse import load_ranked_list_grouped
from list_lib import lmap


def main(input_path):
    claims = get_all_claims()
    claim_d = claims_to_dict(claims)
    gold: Dict[int, List[List[int]]] = get_claim_perspective_id_dict()
    grouped_ranked_list = load_ranked_list_grouped(input_path)

    def is_correct(qid: str, doc_id: str):
        return any([int(doc_id) in cluster for cluster in gold[int(qid)]])

    top_k = 5
    for qid, entries in grouped_ranked_list.items():
        n_gold = sum(map(len, gold[int(qid)]))
        cut_n = min(n_gold, top_k)
        correctness = list([is_correct(qid, e.doc_id) for e in entries[:cut_n]])
        num_correct = sum(lmap(int, correctness))
        p_at_k = num_correct / cut_n

        pid_to_rank: Dict[str, int] = {e.doc_id: e.rank for e in entries}

        def get_rank(pid: int):
            if str(pid) in pid_to_rank:
                return pid_to_rank[str(pid)]
            else:
                return "X"

        if p_at_k < 0.3:
            print(n_gold)
            print(p_at_k)
            print("Claim {} {}".format(qid, claim_d[int(qid)]))##
            for cluster in gold[int(qid)]:
                print("-")
                for pid in cluster:
                    print("[{}]".format(get_rank(pid)), perspective_getter(int(pid)))
            for e in entries[:50]:
                correct_str = "Y" if is_correct(qid, e.doc_id) else "N"
                print("{} {} {}".format(correct_str, e.score, perspective_getter(int(e.doc_id))))



if __name__ == "__main__":
    main(sys.argv[1])