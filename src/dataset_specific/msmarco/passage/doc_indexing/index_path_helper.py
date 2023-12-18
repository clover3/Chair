from dataclasses import dataclass

from omegaconf import OmegaConf

from cpath import output_path, yconfig_dir_path, project_root
from misc_lib import path_join


def get_mmp_working_dir(index_name):
    return path_join(output_path, "mmp", index_name)


def get_mmp_inv_index_path(index_name):
    return path_join(get_mmp_working_dir(index_name), "inv_index")


def get_mmp_df_path(index_name):
    return path_join(get_mmp_working_dir(index_name), "df")


def get_mmp_dl_path(index_name):
    return path_join(get_mmp_working_dir(index_name), "dl")


@dataclass
class BM25IndexResource:
    project_root: str
    index_name: str
    common_dir: str
    inv_index_path: str
    tokenizer: str
    df_path: str
    dl_path: str = ""
    avdl: int = 0


def load_omega_config_with_dataclass(config_path, data_class):
    conf = OmegaConf.structured(data_class)
    conf.merge_with(OmegaConf.load(str(config_path)))
    conf.project_root = project_root
    return conf


def get_bm25_no_stem_resource_path_helper():
    config_path = path_join(
        yconfig_dir_path, "bm25_resource", "no_stem.yaml")
    data_class = BM25IndexResource
    return load_omega_config_with_dataclass(config_path, data_class)


def get_bm25_sp_stem_resource_path_helper():
    config_path = path_join(
        yconfig_dir_path, "bm25_resource", "sp_stem.yaml")
    data_class = BM25IndexResource
    return load_omega_config_with_dataclass(config_path, data_class)


def get_bm25_bert_tokenized_resource_path_helper():
    config_path = path_join(
        yconfig_dir_path, "bm25_resource", "bert_tokenize.yaml")
    data_class = BM25IndexResource
    return load_omega_config_with_dataclass(config_path, data_class)


def get_bm25_bt2_resource_path_helper():
    config_path = path_join(
        yconfig_dir_path, "bm25_resource", "bt2.yaml")
    return load_bm25_index_resource_conf(config_path)


def load_bm25_index_resource_conf(config_path):
    data_class = BM25IndexResource
    return load_omega_config_with_dataclass(config_path, data_class)


def get_bm25_bt3_resource_path_helper():
    config_path = path_join(
        yconfig_dir_path, "bm25_resource", "bt3.yaml")
    data_class = BM25IndexResource
    return load_omega_config_with_dataclass(config_path, data_class)


def get_bm25_stem_resource_path_helper():
    config_path = path_join(
        yconfig_dir_path, "bm25_resource", "stem.yaml")
    data_class = BM25IndexResource
    return load_omega_config_with_dataclass(config_path, data_class)

