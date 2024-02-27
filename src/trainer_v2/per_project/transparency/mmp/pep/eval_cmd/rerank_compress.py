import sys
import sys
from omegaconf import OmegaConf

from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank_w_conf import run_rerank_with_conf_common
from trainer_v2.per_project.transparency.mmp.pep.pep_rerank import get_pep_scorer_from_two_model

from cpath import output_path
from misc_lib import path_join
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def main():
    c_log.info(__file__)
    get_scorer_fn = get_pep_scorer_from_two_model
    dataset_conf_path = sys.argv[1]
    model_name = sys.argv[2]
    step = sys.argv[3]
    model_path = path_join(output_path, "model", "runs", model_name, f"model_{step}")

    run_name = f"{model_name}_{step}"
    conf = OmegaConf.create({
        "run_name": run_name,
        "model_path": model_path,
        "dataset_conf_path": dataset_conf_path,
        "outer_batch_size": 1024,
        "partitioner": "compress"
    })

    strategy = get_strategy()
    with strategy.scope():
        run_rerank_with_conf_common(conf, get_scorer_fn)


if __name__ == "__main__":
    main()
