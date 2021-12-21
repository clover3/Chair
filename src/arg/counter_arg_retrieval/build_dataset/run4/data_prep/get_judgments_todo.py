from typing import Tuple, Dict, List

from arg.counter_arg_retrieval.build_dataset.judgments import Judgement
from arg.counter_arg_retrieval.build_dataset.passage_scoring.split_passages import PassageRange
from arg.counter_arg_retrieval.build_dataset.path_helper import load_sliced_passage_ranked_list
from arg.counter_arg_retrieval.build_dataset.run3.run_interface.split_documents import load_ca3_swtt_passage
from arg.counter_arg_retrieval.build_dataset.run4.run4_util import load_run4_swtt_passage
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from list_lib import index_by_fn
from trec.types import DocID


def document_overlap():
    run_name = "PQ_1"
    all_doc_ids = get_doc_ids_from_run(run_name)
    print(len(all_doc_ids))

    runs = ["PQ_6", "PQ_7", "PQ_8", "PQ_9"]
    for run_name in runs:
        doc_ids = get_doc_ids_from_run(run_name)
        n_common = len(all_doc_ids.intersection(doc_ids))
        print(n_common, len(doc_ids), run_name)


def get_doc_ids_from_run(run_name):
    pq1 = load_sliced_passage_ranked_list(run_name)
    all_doc_ids = set()
    for qid, entries in pq1.items():
        passage_doc_ids = [e.doc_id for e in entries]
        doc_id_list = []
        for p_doc_id in passage_doc_ids:
            doc_id, passage_idx = p_doc_id.split("_")
            doc_id_list.append(doc_id)

        all_doc_ids.update(doc_id_list)
    return all_doc_ids


def get_all_6_to_9():
    runs = ["PQ_6", "PQ_7", "PQ_8", "PQ_9"]
    passages: Dict[DocID, Tuple[SegmentwiseTokenizedText, List[PassageRange]]] \
        = load_run4_swtt_passage()
    judgments_todo = []
    for run_name in runs:
        pq = load_sliced_passage_ranked_list(run_name)
        required_judgements = convert_to_judgment_entries(passages, pq)
        for e in required_judgements:
            if e in judgments_todo:
                pass
            else:
                judgments_todo.append(e)
    return judgments_todo


def check_if_same_passage_idx_makes_same_st_ed():
    prev_judgments = load_prev_judgments()
    prev_judgments_d = index_by_fn(Judgement.get_new_doc_id, prev_judgments)

    cur_judgments = get_all_6_to_9()

    n_equal = 0
    n_wrong = 0
    for e in cur_judgments:
        new_doc_id = e.get_new_doc_id()
        if new_doc_id in prev_judgments_d:
            p_j = prev_judgments_d[new_doc_id]
            if p_j == e:
                n_equal += 1
            else:
                print()
                print(p_j.passage_st, p_j.passage_ed)
                print(e.passage_st, e.passage_ed)
                n_wrong += 1

    print(n_equal, n_wrong)


def get_judgments_todo():
    prev_judgments = load_prev_judgments()
    passages: Dict[DocID, Tuple[SegmentwiseTokenizedText, List[PassageRange]]] \
        = load_run4_swtt_passage()

    n_prev_judged = 0
    runs = ["PQ_6", "PQ_7", "PQ_8", "PQ_9"]
    judgments_todo = []
    for run_name in runs:
        pq = load_sliced_passage_ranked_list(run_name)
        required_judgements = convert_to_judgment_entries(passages, pq)
        print(run_name)
        for e in required_judgements:
            if e in prev_judgments:
                n_prev_judged += 1
                pass
            elif e in judgments_todo:
                pass
            else:
                judgments_todo.append(e)
    print("n_prev_judged", n_prev_judged)
    return judgments_todo


def load_prev_judgments():
    passages: Dict[DocID, Tuple[SegmentwiseTokenizedText, List[PassageRange]]] \
        = load_ca3_swtt_passage()
    run_name = "PQ_1"
    pq1 = load_sliced_passage_ranked_list(run_name)
    prev_judgments = convert_to_judgment_entries(passages, pq1)
    return prev_judgments


def convert_to_judgment_entries(passages, pq1):
    all_judgements = []
    for qid, entries in pq1.items():
        for e in entries:
            doc_id, passage_idx = e.doc_id.split("_")
            passage_idx = int(passage_idx)
            swtt, passage_ranges = passages[doc_id]
            st, ed = passage_ranges[passage_idx]
            judgement = Judgement(qid, doc_id, passage_idx, st, ed)
            all_judgements.append(judgement)
    return all_judgements


def main():
    todo = get_judgments_todo()
    print("{} judgments to do ".format(len(todo)))


if __name__ == "__main__":
    check_if_same_passage_idx_makes_same_st_ed()
