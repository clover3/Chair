from arg.pers_evidence.get_lm import get_query_lms
from arg.qck.decl import QKUnit
from arg.qck.filter_qk import filter_qk
from cache import save_to_pickle, load_from_pickle
from list_lib import lfilter
from misc_lib import tprint


def main():
    qk_list = load_from_pickle("pc_evidence_qk")
    split = "train"
    split = "dev"
    tprint("Building query lms")
    query_lms = get_query_lms(split)
    split_query_ids = list(query_lms.keys())

    def is_split(qk: QKUnit):
        q, k = qk
        if q.query_id in split_query_ids:
            return True
        else:
            return False

    qk_for_split = lfilter(is_split, qk_list)
    tprint("start filtering")
    filtered_qk = filter_qk(qk_for_split, query_lms)

    save_to_pickle(filtered_qk, "pc_evi_filtered_qk_{}".format(split))


if __name__ == "__main__":
    main()







