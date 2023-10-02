import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Iterable, Callable, Dict, Tuple, Set

from cache import load_pickle_from
from omegaconf import OmegaConf

from cpath import yconfig_dir_path, project_root
from misc_lib import exist_or_mkdir, path_join
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import get_fidelity_save_name


@dataclass
class PerCorpusPathConfig:
    project_root: str
    per_corpus_path: str

    frequent_q_terms: str
    tfs_save_path_by_qid: str
    shallow_scores_by_qid: str

    # Per partition (thus differ by train/dev)
    qtfs_path: str


@dataclass
class PerModelPathConfig:
    project_root: str
    per_corpus_path: str
    # Per QID

    deep_scores_by_qid: str


@dataclass
class PerPairCandidates:
    project_root: str
    per_corpus_path: str

    candidate_pair_path: str
    # Save directory
    term_effect_save_dir: str
    fidelity_save_dir: str
    fidelity_table_path: str
    frequent_q_terms: str = ""
    q_term_index_path: str = ""


def path_join(*args) -> pathlib.Path:
    cur_path = pathlib.Path(args[0])
    for item in args[1:]:
        cur_path = cur_path.joinpath(item)
    return cur_path


class MMPGAlignPathHelper:
    def __init__(
            self,
            per_corpus: PerCorpusPathConfig,
            per_model: PerModelPathConfig,
            per_pair_candidates: PerPairCandidates,
    ):
        self.per_corpus = per_corpus
        self.per_model = per_model
        self.per_pair_candidates = per_pair_candidates

    def load_mmp_tfs(self, qid):
        file_path = path_join(self.per_corpus.tfs_save_path_by_qid, str(qid))
        return load_pickle_from(file_path)

    def load_qtfs(self, split, i):
        return self.per_corpus.qtfs_path

    def load_qtfs_indexed(self, split, i) -> Dict[str, List[str]]:
        qid_qtfs = self.load_qtfs(split, i)
        inv_index = defaultdict(list)
        for qid, tfs in qid_qtfs:
            for t in tfs:
                inv_index[t].append(qid)
        return inv_index

    def load_freq_q_terms(self) -> List[str]:
        f = open(pathlib.Path(self.per_corpus.frequent_q_terms), "r")
        return [line.strip() for line in f]

    def load_qterm_candidates(self) -> List[str]:
        f = open(pathlib.Path(self.per_pair_candidates.frequent_q_terms), "r")
        return [line.strip() for line in f]


    def get_fidelity_save_path(self, q_term, d_term):
        save_name = get_fidelity_save_name(q_term, d_term)
        dir_path = self.per_pair_candidates.fidelity_save_dir
        save_path = path_join(dir_path, save_name)
        return save_path

    def get_sub_dir_partition_path(self, dir_name, partition_no):
        path = pathlib.Path(self.per_corpus.per_corpus_path)
        dir_path = path.joinpath(dir_name)
        exist_or_mkdir(dir_path)
        return dir_path.joinpath(str(partition_no))

    def load_candidate_pairs(self):
        per_candidate_conf = self.per_pair_candidates
        term_pairs = [line.strip().split("\t") for line in open(per_candidate_conf.candidate_pair_path, "r")]
        return term_pairs


def load_omega_config_with_dataclass(config_path, data_class):
    conf = OmegaConf.structured(data_class)
    conf.merge_with(OmegaConf.load(str(config_path)))
    conf.project_root = project_root
    return conf



def get_mmp_galign_path_helper(
        per_corpus_config_path,
        per_model_config_path,
        per_candidate_config_path) -> MMPGAlignPathHelper:

    load_config = load_omega_config_with_dataclass
    per_corpus_config = load_config(per_corpus_config_path, PerCorpusPathConfig)
    per_model_config = load_config(per_model_config_path, PerModelPathConfig)
    per_candidates_config = load_config(per_candidate_config_path, PerPairCandidates)

    return MMPGAlignPathHelper(per_corpus_config, per_model_config,
        per_candidates_config)


def get_mmp_train_corpus_config():
    per_corpus_config_path = path_join(yconfig_dir_path, "mmp_train.yaml")
    return load_omega_config_with_dataclass(per_corpus_config_path, PerCorpusPathConfig)


def get_cand2_1_path_helper():
    per_corpus_config_path = path_join(yconfig_dir_path, "mmp_train.yaml")
    per_model_config_path = path_join(yconfig_dir_path, "mmp1.yaml")
    per_candidate_config_path = path_join(yconfig_dir_path, "candidates2_1.yaml")
    path_helper = get_mmp_galign_path_helper(
        per_corpus_config_path,
        per_model_config_path,
        per_candidate_config_path,
    )
    return path_helper


def get_cand2_1_debug_path_helper():
    per_corpus_config_path = path_join(yconfig_dir_path, "mmp_train.yaml")
    per_model_config_path = path_join(yconfig_dir_path, "mmp1.yaml")
    per_candidate_config_path = path_join(yconfig_dir_path, "candidates2_1_debug.yaml")
    path_helper = get_mmp_galign_path_helper(
        per_corpus_config_path,
        per_model_config_path,
        per_candidate_config_path,
    )
    return path_helper


def get_cand2_1_spearman_path_helper():
    per_corpus_config_path = path_join(yconfig_dir_path, "mmp_train.yaml")
    per_model_config_path = path_join(yconfig_dir_path, "mmp1.yaml")
    per_candidate_config_path = path_join(yconfig_dir_path, "candidates2_1_spearman.yaml")
    path_helper = get_mmp_galign_path_helper(
        per_corpus_config_path,
        per_model_config_path,
        per_candidate_config_path,
    )
    return path_helper


def get_cand2_2_path_helper():
    per_corpus_config_path = path_join(yconfig_dir_path, "mmp_train.yaml")
    per_model_config_path = path_join(yconfig_dir_path, "mmp1.yaml")
    per_candidate_config_path = path_join(yconfig_dir_path, "candidates2_2.yaml")
    path_helper = get_mmp_galign_path_helper(
        per_corpus_config_path,
        per_model_config_path,
        per_candidate_config_path,
    )
    return path_helper

def get_cand4_path_helper():
    per_corpus_config_path = path_join(yconfig_dir_path, "mmp_train.yaml")
    per_model_config_path = path_join(yconfig_dir_path, "mmp1.yaml")
    per_candidate_config_path = path_join(yconfig_dir_path, "candidate4.yaml")
    path_helper = get_mmp_galign_path_helper(
        per_corpus_config_path,
        per_model_config_path,
        per_candidate_config_path,
    )
    return path_helper
