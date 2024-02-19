import sys
from omegaconf import OmegaConf
from cpath import yconfig_dir_path
from misc_lib import path_join
from ptorch.cross_encoder.get_ce_msmarco_mini_lm import get_ce_msmarco_mini_lm_score_fn
from adhoc.resource.dataset_conf_helper import get_rerank_dataset_conf_path
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank_w_conf import run_rerank_with_conf2
from utils.conf_helper import unpack_conf


def run(dataset):
    run_name = "ce_mini_lm"
    dataset_conf_path = get_rerank_dataset_conf_path(dataset)
    conf = OmegaConf.create(
        {
            "bm25conf_path": path_join(yconfig_dir_path, "bm25_resource", "lucene_krovetz.yaml"),
            "dataset_conf_path": dataset_conf_path,
            "table_type": "Score",
            "method": run_name,
            "run_name": run_name,
            "outer_batch_size": 64,
        }
    )
    print(unpack_conf(conf))
    score_fn = get_ce_msmarco_mini_lm_score_fn()
    run_rerank_with_conf2(score_fn, conf)


def main():
    dataset = sys.argv[1]
    run(dataset)


if __name__ == "__main__":
    main()