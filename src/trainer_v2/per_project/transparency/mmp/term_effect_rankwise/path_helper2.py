import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Iterable, Callable, Dict, Tuple, Set

from cache import load_pickle_from
from omegaconf import OmegaConf

from cpath import yconfig_dir_path, project_root
from misc_lib import exist_or_mkdir
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import get_fidelity_save_name


@dataclass
class PathConfig:
    project_root: str
    per_corpus_path: str

    candidate_pair: str
    frequent_q_terms: str
    # Per QID
    tfs_save_path_by_qid: str
    shallow_scores_by_qid: str
    deep_scores_by_qid: str

    # Per partition (thus differ by train/dev)
    qtfs_path_train: str
    qtfs_path_dev: str


@dataclass
class TermEffectConfig:
    term_effect_save_dir: str
    fidelity_save_dir: str


@dataclass
class MMPGAlignConfig:
    path_config: PathConfig
    # term_effect_config: TermEffectConfig


def path_join(*args) -> pathlib.Path:
    cur_path = pathlib.Path(args[0])
    for item in args[1:]:
        cur_path = cur_path.joinpath(item)
    return cur_path


class MMPGAlignPathHelper:
    def __init__(self, config: MMPGAlignConfig):
        self.config = config
        self.path_config = config.path_config


    def load_mmp_tfs(self, qid):
        file_path = path_join(self.path_config.tfs_save_path_by_qid, str(qid))
        return load_pickle_from(file_path)

    def load_qtfs(self, split, i):
        if split == "train":
            dir_path = self.path_config.qtfs_path_train
        elif split == "dev":
            dir_path = self.path_config.qtfs_path_dev
        else:
            raise ValueError()
        file_path = path_join(dir_path, str(i))
        return load_pickle_from(file_path)

    def load_qtfs_indexed(self, split, i) -> Dict[str, List[str]]:
        qid_qtfs = self.load_qtfs(split, i)
        inv_index = defaultdict(list)
        for qid, tfs in qid_qtfs:
            for t in tfs:
                inv_index[t].append(qid)
        return inv_index

    def load_freq_q_terms(self):
        f = open(pathlib.Path(self.config.path_config.frequent_q_terms), "r")
        return [line.strip() for line in f]

    def get_fidelity_save_path(self, q_term, d_term):
        save_name = get_fidelity_save_name(d_term, q_term)
        dir_path = self.config.term_effect_config.fidelity_save_dir
        save_path = path_join(dir_path, save_name)
        return save_path

    def get_sub_dir_partition_path(self, dir_name, partition_no):
        path = pathlib.Path(self.config.path_config.per_corpus_path)
        dir_path = path.joinpath(dir_name)
        exist_or_mkdir(dir_path)
        return dir_path.joinpath(str(partition_no))

def get_mmp_galign_path_helper() -> MMPGAlignPathHelper:
    path_config_path = path_join(yconfig_dir_path, "mmp_path_config.yaml")
    conf = OmegaConf.structured(PathConfig)
    conf.merge_with(OmegaConf.load(str(path_config_path)))
    conf.project_root = project_root
    conf = MMPGAlignConfig(conf)
    return MMPGAlignPathHelper(conf)


