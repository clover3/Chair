import sys

from omegaconf import OmegaConf

from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank_w_conf import run_rerank_with_conf_common
from trainer_v2.per_project.transparency.mmp.pep.pep_rerank import get_pep_scorer_from_two_model
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def main():
    c_log.info(__file__)
    get_scorer_fn = get_pep_scorer_from_two_model
    model_name = sys.argv[1]
    step = int(sys.argv[2])
    model_path = f"output/model/runs2/{model_name}/model_{step}"
    mmp, pep_ver = model_name.split("_")
    step_k = step // 1000
    run_name = f"{pep_ver}_{step_k}K"
    print(run_name)
    conf = OmegaConf.create({
        "run_name": run_name,
        "model_path": model_path,
        "dataset_conf_path": "confs/dataset_conf/mmp_dev_sample_C.yaml",
        "outer_batch_size": 1024
    })
    strategy = get_strategy()
    with strategy.scope():
        run_rerank_with_conf_common(conf, get_scorer_fn)


if __name__ == "__main__":
    main()
