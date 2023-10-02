from typing import List, Tuple

from misc_lib import select_first_second, group_by, get_first
from table_lib import tsv_iter
from trec.ranked_list_util import build_ranked_list
from trec.trec_parse import write_trec_ranked_list_entry
from trec.types import TrecRankedListEntry


def build_ranked_list_from_qid_pid_scores(qid_pid_path, run_name, save_path, scores_path):
    qid_pid: List[Tuple[str, str]] = list(select_first_second(tsv_iter(qid_pid_path)))
    scores = read_scores(scores_path)
    items = [(qid, pid, score) for (qid, pid), score in zip(qid_pid, scores)]

    grouped = group_by(items, get_first)
    all_entries: List[TrecRankedListEntry] = []
    for qid, entries in grouped.items():
        scored_docs = [(pid, score) for _, pid, score in entries]
        entries = build_ranked_list(qid, run_name, scored_docs)
        all_entries.extend(entries)
    write_trec_ranked_list_entry(all_entries, save_path)


def read_scores(scores_path):
    scores = []
    for line in open(scores_path, "r"):
        scores.append(float(line))
    return scores