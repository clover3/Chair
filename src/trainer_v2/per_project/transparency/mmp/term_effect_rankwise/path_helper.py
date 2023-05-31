from collections import Counter, defaultdict
from typing import List, Tuple
from typing import List, Iterable, Callable, Dict, Tuple, Set

from cache import load_pickle_from
from cpath import output_path
from dataset_specific.msmarco.passage.passage_resource_loader import enum_grouped2, tsv_iter
from misc_lib import path_join
import os


def get_qtfs_save_path(job_no):
    save_path = path_join("output", "msmarco", "passage", "qtfs", str(job_no))
    return save_path


def tfs_and_scores_path(save_name):
    save_path = path_join("output", "msmarco", "passage", "tfs_and_scores", save_name)
    return save_path


def term_effect_dir():
    save_path = path_join("output", "msmarco", "passage", "term_effect_space")
    return save_path


def get_deep_model_score_path(job_no):
    scores_path = path_join(output_path, "msmarco", "passage", "mmp_train_split_all_scores", f"{job_no}.scores")
    return scores_path


def load_from_sub_files(get_sub_file_path):
    all_cbe = []
    for sub_job_no in range(10000):
        file_path = get_sub_file_path(sub_job_no)
        if os.path.exists(file_path):
            cbe = load_pickle_from(file_path)
            all_cbe.extend(cbe)
        else:
            break
    return all_cbe


def get_tfs_save_path(qid):
    dir_path = path_join(output_path, "msmarco", "passage", "tfs_by_qid")
    save_path = path_join(dir_path, str(qid))
    return save_path


def load_mmp_tfs(qid) -> Tuple[str, List[Tuple[str, Counter]]]:
    return load_pickle_from(get_tfs_save_path(qid))


def get_shallow_score_save_path(job_no):
    dir_path = path_join(output_path, "msmarco", "passage", "shallow_scores")
    save_path = path_join(dir_path, str(job_no))
    return save_path



def get_shallow_score_save_path_by_qid(qid):
    dir_path = path_join(output_path, "msmarco", "passage", "shallow_scores_by_qid")
    save_path = path_join(dir_path, str(qid))
    return save_path


def get_deep_score_save_path_by_qid(qid):
    dir_path = path_join(output_path, "msmarco", "passage", "deep_scores_by_qid")
    save_path = path_join(dir_path, str(qid))
    return save_path


def load_tfs_and_computed_base_scores(job_no) -> List[List[Tuple[str, Counter, float]]]:
    dir_path = path_join(output_path, "msmarco", "passage", "tfs_and_scores")
    def get_sub_file_path(sub_job_no):
        return path_join(dir_path, f"{job_no}_{sub_job_no}")
    all_cbe: List[List[Tuple[str, Counter, float]]] = load_from_sub_files(get_sub_file_path)
    return all_cbe


def load_qtfs(job_no) -> List[Tuple[str, Counter]]:
    return load_pickle_from(get_qtfs_save_path(job_no))


def load_qtf_index(job_no) -> Dict[str, List[str]]:
    inv_index = defaultdict(list)
    qid_qtfs = load_qtfs(job_no)
    for qid, tfs in qid_qtfs:
        for t in tfs:
            inv_index[t].append(qid)

    n_qids = len(qid_qtfs)
    # too_frequent_terms = []
    # for t, entries in inv_index.items():
    #     appear_rate = len(entries) / n_qids
    #     if appear_rate > 0.5:
    #         too_frequent_terms.append(appear_rate)
    return inv_index


def read_shallow_scores(job_id) -> List[Tuple[str, List[Tuple[str, float]]]]:
    save_path = get_shallow_score_save_path(job_id)
    output: List[Tuple[str, List[Tuple[str, float]]]] = []
    for group in enum_grouped2(tsv_iter(save_path)):
        qid, _, _ = group[0]
        entries = []
        for qid_, pid, score in group:
            entries.append((pid, float(score)))
        output.append((qid, entries))
    return output


def get_te_save_path(q_term, d_term, job_no):
    save_name = f"{q_term}_{d_term}_{job_no}.jsonl"
    save_path = path_join(term_effect_dir(), save_name)
    return save_path


def read_shallow_score_per_qid(qid) -> Tuple[str, List[Tuple[str, float]]]:
    save_path = get_shallow_score_save_path_by_qid(qid)
    entries = []
    for pid, score in tsv_iter(save_path):
        entries.append((pid, float(score)))
    return qid, entries


def read_deep_score_per_qid(qid) -> Tuple[str, List[Tuple[str, float]]]:
    save_path = get_deep_score_save_path_by_qid(qid)
    entries = []
    for qid, pid, score in tsv_iter(save_path):
        entries.append((pid, float(score)))
    return qid, entries
