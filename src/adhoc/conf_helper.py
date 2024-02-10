from dataclasses import dataclass
from typing import Optional

from omegaconf import OmegaConf

from cpath import project_root


@dataclass
class BM25IndexResource:
    project_root: Optional[str]
    index_name: str
    common_dir: str
    tokenizer: str
    inv_index_path: str
    df_path: str
    dl_path: str
    stopword_path: Optional[str]

    avdl: int = 52
    k1: float = 1.2
    k2: float = 100
    b: float = 0.75


@dataclass
class QLIndexResource:
    project_root: Optional[str]
    index_name: str
    common_dir: str
    tokenizer: str
    inv_index_path: str
    bg_prob_path: str
    dl_path: str
    stopword_path: Optional[str]
    mu: float = 2500

def load_omega_config(
        config_path,
        data_class=None,
        set_project_root=False):

    raw_conf = OmegaConf.load(str(config_path))
    if data_class is None:
        conf = raw_conf
    else:
        conf = OmegaConf.structured(data_class)
        conf.merge_with(raw_conf)

    if set_project_root:
        conf.project_root = project_root
    return conf
