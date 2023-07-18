from collections import Counter, defaultdict
from typing import List, Dict, Tuple

from cache import load_pickle_from
from cpath import output_path
from dataset_specific.msmarco.passage.passage_resource_loader import enum_grouped2, tsv_iter
from misc_lib import path_join
import os

from trainer_v2.per_project.transparency.misc_common import load_str_float_tsv
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.parse_helper import read_qid_pid_score_tsv, escape


def mmp_root():
    return path_join(output_path, "msmarco", "passage")


def get_qtfs_save_path_train(job_no):
    save_path = path_join(mmp_root(), "qtfs", str(job_no))
    return save_path


def get_qtfs_save_path(split, job_no):
    if split == "train":
        return get_qtfs_save_path_train(job_no)
    else:
        save_path = path_join(mmp_root(), f"{split}_qtfs", str(job_no))
    return save_path


def tfs_and_scores_path_train(save_name):
    save_path = path_join(mmp_root(), "tfs_and_scores", save_name)
    return save_path


def tfs_and_scores_path(split, save_name):
    save_path = path_join(mmp_root(), f"tfs_and_scores_{split}", save_name)
    return save_path


def term_effect_dir():
    save_path = path_join(mmp_root(), "term_effect_space")
    return save_path


def get_deep_model_score_path_train(job_no):
    scores_path = path_join(mmp_root(), "mmp_train_split_all_scores", f"{job_no}.scores")
    return scores_path


def get_deep_model_score_path(split, job_no):
    scores_path = path_join(mmp_root(),
                            f"mmp_{split}_split_all_scores", f"{job_no}.scores")
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
    dir_path = path_join(mmp_root(), "tfs_by_qid")
    save_path = path_join(dir_path, str(qid))
    return save_path


def load_mmp_tfs(qid) -> Tuple[str, List[Tuple[str, Counter]]]:
    return load_pickle_from(get_tfs_save_path(qid))


def get_shallow_score_save_path_train(job_no):
    dir_path = path_join(mmp_root(), "shallow_scores")
    save_path = path_join(dir_path, str(job_no))
    return save_path


def get_shallow_score_save_path_dev(job_no):
    dir_path = path_join(mmp_root(), "shallow_scores_dev")
    save_path = path_join(dir_path, str(job_no))
    return save_path


def get_shallow_score_save_path(split, job_no):
    if split == "train":
        return get_shallow_score_save_path_train(job_no)
    elif split == "dev":
        return get_shallow_score_save_path_dev(job_no)
    else:
        raise ValueError()


def get_shallow_score_save_path_by_qid(qid):
    dir_path = path_join(mmp_root(), "shallow_scores_by_qid")
    save_path = path_join(dir_path, str(qid))
    return save_path


def get_deep_score_save_path_by_qid(qid):
    dir_path = path_join(mmp_root(), "deep_scores_by_qid")
    save_path = path_join(dir_path, str(qid))
    return save_path


def get_te_save_path2(q_term, d_term, job_no):
    save_name = get_te_save_name(q_term, d_term, job_no)
    dir_path = path_join(mmp_root(), "term_effect_space2", "content")
    save_path = path_join(dir_path, save_name)
    return save_path


def get_te_save_name(q_term, d_term, job_no):
    q_term_es = escape(q_term)
    d_term_es = escape(d_term)
    save_name = f"{q_term_es}_{d_term_es}_{job_no}.jsonl.gz"
    return save_name


def get_fidelity_save_path2(q_term, d_term):
    save_name = get_fidelity_save_name(q_term, d_term)
    dir_path = path_join(mmp_root(), "term_effect_space2", "fidelity")
    save_path = path_join(dir_path, save_name)
    return save_path


def get_fidelity_save_name(q_term, d_term):
    q_term_es = escape(q_term)
    d_term_es = escape(d_term)
    save_name = f"{q_term_es}_{d_term_es}.txt"
    return save_name


def term_align_candidate2_score_path():
    save_path = path_join(
        mmp_root(), "align_scores", "candidate2.tsv")
    return save_path


def get_te_save_path_base(q_term, d_term, job_no):
    save_name = f"{q_term}_{d_term}_{job_no}.jsonl"
    save_path = path_join(term_effect_dir(), save_name)
    return save_path


def load_tfs_and_computed_base_scores(job_no) -> List[List[Tuple[str, Counter, float]]]:
    dir_path = path_join(mmp_root(), "tfs_and_scores")

    def get_sub_file_path(sub_job_no):
        return path_join(dir_path, f"{job_no}_{sub_job_no}")
    all_cbe: List[List[Tuple[str, Counter, float]]] = load_from_sub_files(get_sub_file_path)
    return all_cbe


def load_qtfs_train(job_no) -> List[Tuple[str, Counter]]:
    return load_pickle_from(get_qtfs_save_path_train(job_no))


def load_qtfs(split, job_no) -> List[Tuple[str, Counter]]:
    return load_pickle_from(get_qtfs_save_path(split, job_no))


def load_qtf_index_train(job_no) -> Dict[str, List[str]]:
    inv_index = defaultdict(list)
    qid_qtfs = load_qtfs_train(job_no)
    for qid, tfs in qid_qtfs:
        for t in tfs:
            inv_index[t].append(qid)

    return inv_index


def load_qtf_index_from_qid_qtfs(save_path) -> Dict[str, List[str]]:
    inv_index = defaultdict(list)
    qid_qtfs = load_pickle_from(save_path)
    for qid, tfs in qid_qtfs:
        for t in tfs:
            inv_index[t].append(qid)

    return inv_index


def read_shallow_scores_train(job_id) -> List[Tuple[str, List[Tuple[str, float]]]]:
    save_path = get_shallow_score_save_path_train(job_id)
    return read_qid_pid_score_triplet_grouped(save_path)


def read_qid_pid_score_triplet_grouped(save_path):
    output: List[Tuple[str, List[Tuple[str, float]]]] = []
    for group in enum_grouped2(tsv_iter(save_path)):
        qid, _, _ = group[0]
        entries = []
        for qid_, pid, score in group:
            entries.append((pid, float(score)))
        output.append((qid, entries))
    return output


def read_shallow_score_per_qid(qid) -> Tuple[str, List[Tuple[str, float]]]:
    save_path = get_shallow_score_save_path_by_qid(qid)
    return load_str_float_tsv(qid, save_path)


def read_deep_score_per_qid(qid) -> Tuple[str, List[Tuple[str, float]]]:
    save_path = get_deep_score_save_path_by_qid(qid)
    return read_qid_pid_score_tsv(qid, save_path)


