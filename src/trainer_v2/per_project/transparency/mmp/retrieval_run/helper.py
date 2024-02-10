from cpath import yconfig_dir_path
from misc_lib import path_join


def get_dataset_conf_path(dataset):
    dataset_conf_path = {
        "trec_dl19": path_join(yconfig_dir_path, "dataset_conf", "retrieval_trec_dl_2019_43.yaml"),
        "trec_dl20": path_join(yconfig_dir_path, "dataset_conf", "retrieval_trec_dl_2020.yaml"),
        "dev_c": path_join(yconfig_dir_path, "dataset_conf", "retrieval_mmp_dev_C.yaml"),
        "dev1000": path_join(yconfig_dir_path, "dataset_conf", "retrieval_mmp_dev1000.yaml"),
    }[dataset.lower()]
    return dataset_conf_path
