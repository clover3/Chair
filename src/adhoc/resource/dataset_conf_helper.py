import os

from omegaconf import OmegaConf

from adhoc.conf_helper import load_omega_config
from cpath import yconfig_dir_path
from misc_lib import path_join
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank_w_conf import RerankDatasetConf


def get_rerank_dataset_conf_path(dataset):
    dataset_conf_path = {
        "trec_dl19": path_join(yconfig_dir_path, "dataset_conf", "trec_dl_2019.yaml"),
        "trec_dl20": path_join(yconfig_dir_path, "dataset_conf", "trec_dl_2020.yaml"),
        "dev_c": path_join(yconfig_dir_path, "dataset_conf", "mmp_dev_sample_C.yaml"),
        "dev1000": path_join(yconfig_dir_path, "dataset_conf", "mmp_dev_sample1000.yaml"),
        "mmp_train_sample": path_join(yconfig_dir_path, "dataset_conf", "mmp_train_sample.yaml"),
        "mmp_dev_sample1k_a": path_join(yconfig_dir_path, "dataset_conf", "mmp_dev_sample1K_A.yaml"),
    }[dataset.lower()]
    return dataset_conf_path


def get_beir_dataset_conf_path(dataset):
    conf_path = path_join(yconfig_dir_path, "dataset_conf", f"rr_{dataset}.yaml")
    return conf_path


def build_dev1K_A_conf(dataset):
    sig = "dev1K_A_"
    partition_no = int(dataset[len(sig):])

    def reform(file_path):
        file_name, file_extension = os.path.splitext(file_path)
        return f"{file_name}_{partition_no}{file_extension}"

    conf_path = path_join(yconfig_dir_path, "dataset_conf", "mmp_dev_sample1K_A.yaml")
    conf = load_omega_config(conf_path, RerankDatasetConf)
    conf.rerank_payload_path = reform(conf.rerank_payload_path)
    conf.dataset_name = dataset
    conf.data_size = 100000
    return conf


def get_dataset_conf(dataset) -> RerankDatasetConf:
    from dataset_specific.beir_eval.beir_common import beir_dataset_list_A
    if dataset in beir_dataset_list_A:
        conf_path = get_beir_dataset_conf_path(dataset)
        conf = load_omega_config(conf_path, RerankDatasetConf)
    elif dataset.startswith("dev1K_A_"):
        conf = build_dev1K_A_conf(dataset)
    else:
        conf_path = get_rerank_dataset_conf_path(dataset)
        conf = load_omega_config(conf_path, RerankDatasetConf)
    return conf
