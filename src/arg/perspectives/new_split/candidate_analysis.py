import json
import os
from collections import defaultdict, Counter
from typing import List, Iterable, Dict, Tuple, Set

from arg.perspectives.load import splits
from cpath import output_path, data_path
from evals.parse import load_qrels_flat
from trec.trec_parse import load_ranked_list_grouped, TrecRankedListEntry


def do_for_perspectrum():
    # Count candidates that appear as positive in training split but negative in dev/test
    judgment_path = os.path.join(data_path, "perspective", "qrel.txt")
    qrels = load_qrels_flat(judgment_path)

    candidate_set_name = "pc_qres"
    # candidate_set_name = "default_qres"

    def get_ranked_list(split):
        ranked_list_path = os.path.join(output_path,
                                        "perspective_experiments",
                                        candidate_set_name, "{}.txt".format(split))
        rlg: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranked_list_path)
        return rlg

    train_splits = ["train"]
    eval_splits = ["dev"]
    analyze_overlap(get_ranked_list, qrels, train_splits, eval_splits)


def load_qid_for_split(split_d, split) -> Iterable[str]:
    for qid in split_d:
        if split_d[qid] == split:
            yield qid


def do_for_new_perspectrum():
    judgment_path = os.path.join(data_path, "perspective", "qrel.txt")
    qrels = load_qrels_flat(judgment_path)

    candidate_set_name = "pc_qres"
    # candidate_set_name = "default_qres"

    def get_old_ranked_list(split):
        ranked_list_path = os.path.join(output_path,
                                        "perspective_experiments",
                                        candidate_set_name, "{}.txt".format(split))
        rlg: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranked_list_path)
        return rlg

    rlg_all: Dict[str, List[TrecRankedListEntry]] = dict()
    for split in splits:
        rlg_all.update(get_old_ranked_list(split))

    split_info_path = os.path.join(data_path, "perspective", "new_split.json")
    new_splits: Dict[str, str] = json.load(open(split_info_path, "r"))

    def get_new_ranked_list(split):
        qids: Iterable[str] = load_qid_for_split(new_splits, split)
        new_rlg = {}
        for qid in qids:
            new_rlg[qid] = rlg_all[qid]
        return new_rlg

    train_splits = ["train"]
    eval_splits = ["dev"]
    analyze_overlap(get_new_ranked_list, qrels, train_splits, eval_splits)


def do_for_robust():
    # Count candidates that appear as positive in training split but negative in dev/test
    judgment_path = os.path.join(data_path, "robust", "qrels.rob04.txt")
    qrels = load_qrels_flat(judgment_path)
    qrels['672'] = []

    ranked_list_path = os.path.join(data_path, "robust", "rob04.desc.galago.2k.out")
    rlg_all: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranked_list_path)

    def is_in_split(split, qid):
        if split == "train":
            return int(qid) <= 650
        elif split == "dev":
            return int(qid) >= 651
        else:
            assert False

    def get_ranked_list(split):
        out_rlg = {}
        for qid, rl in rlg_all.items():
            if is_in_split(split, qid):
                out_rlg[qid] = rl[:100]
        return out_rlg

    train_splits = ["train"]
    eval_splits = ["dev"]
    analyze_overlap(get_ranked_list, qrels, train_splits, eval_splits)

    # data_split: Dict[int, str] = load_data_set_split()
    # claim_d = get_all_claim_d()
    # for doc_id in all_unique_docs:
    #     rel_q_id_list = rev_map[doc_id]
    #     if len(rel_q_id_list) > 1:
    #         splits_appeared = set()
    #         for qid in rel_q_id_list:
    #             splits_appeared.add(data_split[qid])
    #
    #         p_text = perspective_getter(int(doc_id))
    #         print(p_text)
    #         for qid in rel_q_id_list:
    #             claim_text = claim_d[int(qid)]
    #             print("[{}]".format(data_split[qid]), qid, claim_text)
    #


def analyze_overlap(get_ranked_list, qrels, train_splits, eval_splits):
    def get_unique_docs(split):
        doc_id_set = set()
        rlg: Dict[str, List[TrecRankedListEntry]] = get_ranked_list(split)
        for _, rl in rlg.items():
            doc_id_set.update([e.doc_id for e in rl])
        return doc_id_set

    print("num unique docs")
    all_unique_docs = set()
    for split in train_splits + eval_splits:
        per_split_unique_docss: Set[str] = get_unique_docs(split)
        print(split, len(per_split_unique_docss))
        all_unique_docs.update(per_split_unique_docss)
    print("all_unique docs: ", len(all_unique_docs))

    def get_reverse_map() -> Dict[str, List[str]]:
        rev_map = defaultdict(list)
        for qid, gold_list in qrels.items():
            true_gold: List[str] = list([doc_id for doc_id, score in gold_list if score > 0])
            for doc_id in true_gold:
                rev_map[doc_id].append(qid)
        return rev_map

    doc_id_to_qid = defaultdict(list)
    split = "train"
    rlg: Dict[str, List[TrecRankedListEntry]] = get_ranked_list(split)
    train_known_count_pos = Counter()
    train_known_count_neg = Counter()
    for query_id, rl in rlg.items():
        gold_list: List[Tuple[str, int]] = qrels[query_id]
        true_gold: List[str] = list([doc_id for doc_id, score in gold_list if score > 0])
        for e in rl:
            label = e.doc_id in true_gold
            if label:
                doc_id_to_qid[e.doc_id].append(query_id)
                train_known_count_pos[e.doc_id] += 1
            else:
                train_known_count_neg[e.doc_id] += 1
    # for split in ["dev", "test"]:
    for split in eval_splits:
        rlg: Dict[str, List[TrecRankedListEntry]] = get_ranked_list(split)
        count_pos = Counter()
        count_neg = Counter()
        for query_id, rl in rlg.items():
            gold_list: List[Tuple[str, int]] = qrels[query_id]
            true_gold: List[str] = list([doc_id for doc_id, score in gold_list if score > 0])
            for e in rl:
                label = e.doc_id in true_gold
                if label:
                    count_pos[e.doc_id] += 1
                else:
                    count_neg[e.doc_id] += 1
        doc_id_set = set()
        doc_id_set.update(count_pos.keys())
        doc_id_set.update(count_neg.keys())
        any_observed_doc = 0
        observation_combination = Counter()
        for doc_id in doc_id_set:
            if doc_id in train_known_count_pos or doc_id in train_known_count_neg:
                any_observed_doc += 1

            for label_train_appear in [0, 1]:
                for label_eval_appear in [0, 1]:
                    d_train: Counter = [train_known_count_neg, train_known_count_pos][label_train_appear]
                    d_eval: Counter = [count_neg, count_pos][label_eval_appear]
                    if doc_id in d_train and doc_id in d_eval:
                        key = label_train_appear, label_eval_appear
                        observation_combination[key] += 1

        print(split)
        print("any_observed_doc", any_observed_doc)
        for key, cnt in observation_combination.items():
            print(key, cnt)


if __name__ == "__main__":
    do_for_new_perspectrum()