from typing import List, Tuple

from adhoc.eval_helper.pytrec_helper import eval_by_pytrec_json_qrel, eval_by_pytrec
from cpath import output_path

from misc_lib import select_first_second, group_by, get_first, path_join
from table_lib import tsv_iter
from taskman_client.task_proxy import get_task_manager_proxy
from trec.ranked_list_util import build_ranked_list
from trec.trec_parse import write_trec_ranked_list_entry
from trec.types import TrecRankedListEntry


def build_ranked_list_from_qid_pid_scores(qid_pid_path, run_name, save_path, scores_path):
    qid_pid: List[Tuple[str, str]] = list(select_first_second(tsv_iter(qid_pid_path)))
    scores = read_scores(scores_path)
    all_entries = build_rankd_list_from_qid_pid_scores_inner(qid_pid, run_name, scores)
    write_trec_ranked_list_entry(all_entries, save_path)


def build_rankd_list_from_qid_pid_scores_inner(qid_pid, run_name, scores):
    items = [(qid, pid, score) for (qid, pid), score in zip(qid_pid, scores)]
    grouped = group_by(items, get_first)
    all_entries: List[TrecRankedListEntry] = []
    for qid, entries in grouped.items():
        scored_docs = [(pid, score) for _, pid, score in entries]
        entries = build_ranked_list(qid, run_name, scored_docs)
        all_entries.extend(entries)
    return all_entries


def read_scores(scores_path):
    scores = []
    for line in open(scores_path, "r"):
        try:
            s = float(line)
        except ValueError:
            s = eval(line)[0]
        scores.append(s)
    return scores


def build_ranked_list_from_line_scores_and_eval(
        run_name, dataset_name, judgment_path, quad_tsv_path, scores_path,
        metric, do_not_report=False):
    """
    Use line scores to generate TREC style ranked list
    :param run_name:
    :param dataset_name:
    :param judgment_path:
    :param quad_tsv_path:
    :param scores_path:
    :param metric:
    :return:
    """
    ranked_list_path = path_join(output_path, "ranked_list", f"{run_name}_{dataset_name}.txt")
    build_ranked_list_from_qid_pid_scores(
        quad_tsv_path,
        run_name,
        ranked_list_path,
        scores_path)

    ret = eval_by_pytrec(
        judgment_path,
        ranked_list_path,
        metric)

    print(f"{metric}:\t{ret}")
    if not do_not_report:
        proxy = get_task_manager_proxy()
        proxy.report_number(run_name, ret, dataset_name, metric)
