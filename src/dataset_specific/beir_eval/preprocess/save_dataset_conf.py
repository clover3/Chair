from dataset_specific.beir_eval.beir_common import beir_dataset_list_A, get_beir_qrel_path
from dataset_specific.beir_eval.preprocess.build_rerank_payload import build_beir_rerank_conf
import sys
from omegaconf import OmegaConf
from cpath import output_path, yconfig_dir_path
from misc_lib import path_join


def main():
    for dataset in beir_dataset_list_A:
        conf = build_beir_rerank_conf(dataset)
        try:
            data_size = sum([1 for _ in open(conf.rerank_payload_path, "r")])
            conf.data_size = data_size
            conf.judgment_path = path_join("data", "beir", "datasets", dataset, "qrels", f"test.tsv")
            conf_path = path_join(yconfig_dir_path, "dataset_conf", f"rr_{dataset}.yaml")
            OmegaConf.save(conf, open(conf_path, "w"))
        except FileNotFoundError as e:
            print(e)



if __name__ == "__main__":
    main()