from typing import Dict, Tuple

from typing import Dict, Tuple
from typing import List

from arg.perspectives.bm25_predict import get_bm25_module
from arg.perspectives.load import get_claim_perspective_id_dict
from arg.perspectives.load import get_claims_from_ids, load_dev_claim_ids
from arg.perspectives.random_walk.pc_predict import pc_predict_from_vector_query
from arg.perspectives.relevance_based_predictor import predict_from_dict
from cache import load_from_pickle
# num_doc = 541
# avdl = 11.74
from list_lib import left


# num_doc = 541
# avdl = 11.74


def compare_two_runs(pred1: List[Tuple[str, List[Dict]]],
                     pred2: List[Tuple[str, List[Dict]]]):
    gold = get_claim_perspective_id_dict()
    top_k = 7
    claim_ids_list = left(pred1)

    pred1_d = dict(pred1)
    pred2_d = dict(pred2)

    for claim_id in claim_ids_list:
        pers_list_1 = pred1_d[claim_id][:top_k]
        pers_list_2 = pred2_d[claim_id][:top_k]
        gold_pids = gold[claim_id]

        pid_rank_1 = {p['pid']: idx for idx, p in enumerate(pred1_d[claim_id])}
        pid_rank_2 = {p['pid']: idx for idx, p in enumerate(pred2_d[claim_id])}

        def is_correct(pid):
            for pids in gold_pids:
                if pid in pids:
                    return True
            return False

        claim_text = pers_list_1[0]['claim_text']
        pid_to_text = {p['pid']: p['perspective_text'] for p in pers_list_1}
        pid_to_text.update({p['pid']: p['perspective_text'] for p in pers_list_2})

        pids_1 = [p['pid'] for p in pers_list_1]
        pids_2 = [p['pid'] for p in pers_list_2]

        rationale_2 = {p['pid']: p['rationale'] for p in pers_list_2}
        p2_d_all= {p['pid']: p for p in pred2_d[claim_id]}

        all_pids = set(pids_1 + pids_2)

        both_correct = [p for p in all_pids if is_correct(p) and p in pids_1 and p in pids_2]
        only1_correct = [p for p in all_pids if is_correct(p) and p in pids_1 and p not in pids_2]
        only2_correct = [p for p in all_pids if is_correct(p) and p not in pids_1 and p in pids_2]
        both_wrong = [p for p in all_pids if not is_correct(p) and p in pids_1 and p in pids_2]
        print()
        print("Claim: ", claim_text)

        if not only1_correct and not only2_correct:
            continue

        if both_correct:
            print("Both correct >")
            for p in both_correct:
                print(pid_to_text[p])

        if only1_correct:
            print("Only 1 correct >")
            for p in only1_correct:
                print("{} ({})".format(pid_to_text[p], pid_rank_2[p]))

        if only2_correct:
            print("Only 2 correct >")
            for p in only2_correct:
                print("{} ({})".format(pid_to_text[p], pid_rank_1[p]), p2_d_all[p]['score'], rationale_2[p])
            print("2 Got wrong while 1 got correct")
            for p in only1_correct:
                try:
                    print(pid_to_text[p], p2_d_all[p]['score'], p2_d_all[p]['rationale'])
                except:
                    pass

        if both_wrong:
            print("Both wrong > ")
            for p in both_wrong:
                print(pid_to_text[p])

        # 1. Both got correct
        # 2. only one of them correct
        # 3. both wrong


def main():
    d_ids: List[int] = list(load_dev_claim_ids())
    claims = get_claims_from_ids(d_ids)
    top_k = 50
    q_tf_replace = dict(load_from_pickle("random_walk_score_100"))
    bm25 = get_bm25_module()
    pred2 = pc_predict_from_vector_query(bm25, q_tf_replace, claims, top_k)
    pc_score_d = load_from_pickle("pc_bert_baseline_score_d")
    pred1 = predict_from_dict(pc_score_d, claims, top_k)

    compare_two_runs(pred1, pred2)


if __name__ == "__main__":
    main()
