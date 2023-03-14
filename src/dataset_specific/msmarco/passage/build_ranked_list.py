from typing import List, Iterable, Callable, Dict, Tuple, Set
from cpath import output_path, at_output_dir
from misc_lib import path_join, select_first_second, group_by, get_first

from dataset_specific.msmarco.passage.passage_resource_loader import load_qrel, tsv_iter
from trec.ranked_list_util import build_ranked_list
from trec.trec_parse import write_trec_ranked_list_entry
from trec.types import TrecRankedListEntry, QRelsDict


def build_ranked_list_from_qid_pid_scores(qid_pid, run_name, save_path, scores_path):
    scores = []
    for line in open(scores_path, "r"):
        scores.append(float(line))
    items = [(qid, pid, score) for (qid, pid), score in zip(qid_pid, scores)]
    grouped = group_by(items, get_first)
    all_entries: List[TrecRankedListEntry] = []
    for qid, entries in grouped.items():
        scored_docs = [(pid, score) for _, pid, score in entries]
        entries = build_ranked_list(qid, run_name, scored_docs)
        all_entries.extend(entries)
    write_trec_ranked_list_entry(all_entries, save_path)


def main():
    run_name = "splade"
    scores_path = at_output_dir("lines_scores", "splade_dev_sample.txt")
    qid_pid_path = path_join("data", "msmarco", "sample_dev", "corpus.tsv")
    qid_pid: List[Tuple[str, str]] = list(select_first_second(tsv_iter(qid_pid_path)))
    save_path = at_output_dir("ranked_list", "splade_mmp_dev_sample.txt")

    build_ranked_list_from_qid_pid_scores(qid_pid, run_name, save_path, scores_path)

if __name__ == "__main__":
    main()

