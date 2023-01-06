from collections import defaultdict
from typing import List

from dataset_specific.scitail import load_scitail_structured
from trainer_v2.per_project.tli.scitail_qa_eval.eval_helper import load_scores
from trainer_v2.per_project.tli.scitail_qa_eval.path_helper import get_score_save_path, get_ranked_list_save_path
from trec.ranked_list_util import assign_rank
from trec.trec_parse import write_trec_ranked_list_entry
from trec.types import TrecRankedListEntry


def build_qid_d(split):
    seen = set()
    qid_d = {}
    for idx, e in enumerate(load_scitail_structured(split)):
        if e.question not in seen:
            qid_d[e.question] = f"{split}_{idx}"
            seen.add(e.question)
    return qid_d


def convert_inner(split, problems, qid_d, run_name, scores):
    assert len(problems) == len(scores)
    dummy_rank = 0
    qid_grouped = defaultdict(list)
    for idx, (score, problem) in enumerate(zip(scores, problems)):
        qid = qid_d[problem.question]
        doc_id = f"{split}_{idx}"
        e = TrecRankedListEntry(qid, doc_id, dummy_rank, score, run_name)
        qid_grouped[qid].append(e)
    out_entries = []
    for qid, entries in qid_grouped.items():
        out_entries.extend(assign_rank(entries))
    return out_entries


def convert(run_name, split):
    save_name = f"{run_name}_{split}"
    save_path = get_score_save_path(save_name)
    problems = load_scitail_structured(split)
    scores: List[float] = load_scores(save_path)
    qid_d = build_qid_d(split)
    out_entries = convert_inner(split, problems, qid_d, run_name, scores)

    ranked_list_save_path = get_ranked_list_save_path(save_name)
    write_trec_ranked_list_entry(out_entries, ranked_list_save_path)


def main():
    run_name_list = [
        "bm25_clue",
        "bm25_tuned",
        "nli_rev_direct",
        "scitail_rev_direct",
        "nli_pep",
        "nli_pep_idf"
    ]
    run_name_list = [
        "tnli2",
    ]
    for run_name in run_name_list:
        for split in ["dev", "test"]:
            try:
                convert(run_name, split)
            except FileNotFoundError as e:
                print(e)


if __name__ == "__main__":
    main()