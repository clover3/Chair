from collections import Counter, defaultdict
from typing import List, Tuple
from typing import List, Iterable, Callable, Dict, Tuple, Set

from cache import load_pickle_from
from cpath import output_path
from misc_lib import path_join
import os


def get_qtfs_save_path(job_no):
    save_path = path_join("output", "msmarco", "passage", "qtfs", str(job_no))
    return save_path


def tfs_and_scores_path(save_name):
    save_path = path_join("output", "msmarco", "passage", "tfs_and_scores", save_name)
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

