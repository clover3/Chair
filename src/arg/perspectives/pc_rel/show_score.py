from typing import Dict, Tuple, List

from scipy.special import softmax

from arg.perspectives.load import get_claim_perspective_id_dict
from arg.perspectives.pc_rel.collect_score import collect_pc_rel_score
from arg.perspectives.types import CPIDPair, Logits
from cache import load_from_pickle, save_to_pickle
from cpath import pjoin, output_path
from list_lib import flatten
from misc_lib import tprint, TimeEstimator, average


class TwoStepDict:
    def __init__(self, d):
        self.high_d = {}
        for key, value in d.items():
            group_no, sub_no = self.get_sub_keys(key)

            if group_no not in self.high_d:
                self.high_d[group_no] = {}
            self.high_d[group_no][sub_no] = value

    def __getitem__(self, key):
        group_no, sub_no = self.get_sub_keys(key)
        try:
            tprint("lookup1")
            sub_d = self.high_d[group_no]
            tprint("lookup2")
            data = sub_d[sub_no]
            tprint("lookup3")
        except KeyError:
            raise

        return data

    def get_sub_keys(self, key):
        group_no = int(key / 100000)
        sub_no = key % 100000
        return group_no, sub_no



def collect_save_relevance_score():
    prediction_file = pjoin(output_path, "pc_rel")

    info_d = load_from_pickle("pc_rel_info_all")
    print("Building twostepdict")
    #two_step_d = TwoStepDict(info_d)

    # info_list = list(info_d.items())
    # info_list.sort(key=lambda x: x[0])
    # idx = 0
    # for a, b in info_list:
    #     print(a)
    #     assert idx == a
    #     idx += 1
    print("Collect pc_rel")

    relevance_scores: Dict[CPIDPair, List[Tuple[Logits, Logits]]] = collect_pc_rel_score(prediction_file, info_d)
    save_to_pickle(relevance_scores, "pc_relevance_score")


def main():
    relevance_scores: Dict[CPIDPair, List[Tuple[Logits, Logits]]] = load_from_pickle("pc_relevance_score")
    gold = get_claim_perspective_id_dict()

    true_feature = []
    false_feature = []

    ticker = TimeEstimator(len(relevance_scores))
    for key in relevance_scores:
        ticker.tick()
        cid, pid = key

        gold_pids = flatten(gold[cid])
        gold_pids = list([int(pid) for pid in gold_pids])
        correct = pid in gold_pids
        scores: List[Tuple[List[float], List[float]]] = relevance_scores[key]

        c_count = 0
        p_count = 0
        pc_count = 0
        for c_logits, p_logits in scores:
            c_rel = softmax(c_logits)[1] > 0.5
            p_rel = softmax(p_logits)[1] > 0.5

            c_count += int(c_rel)
            p_count += int(p_rel)
            pc_count += int(c_rel and p_rel)

        if correct:
            true_feature.append(pc_count)
        else:
            false_feature.append(pc_count)

    all_feature = true_feature + false_feature
    all_feature.sort()
    mid = int(len(all_feature)/2)
    cut_off = all_feature[mid]

    tp = sum([int(t > cut_off) for t in true_feature])
    fp = sum([int(t > cut_off) for t in false_feature])
    tn = sum([int(t <= cut_off) for t in false_feature])
    fn = sum([int(t <= cut_off) for t in true_feature])

    print(tp, fp, tn, fn)
    print("true feature", average(true_feature))
    print("false feature", average(false_feature))


if __name__ == "__main__":
    #collect_save_relevance_score()
    main()
