import os.path

from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank_w_conf import run_rerank_with_conf2
import sys
from omegaconf import OmegaConf
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank import get_scorer_tf_load_model

from trainer_v2.train_util.get_tpu_strategy import get_strategy


def main():
    dataset_conf_path = sys.argv[1]
    model_path = sys.argv[2]
    run_name = os.path.basename(model_path)

    c_log.info("Building scorer")
    max_seq_len = 512

    strategy = get_strategy()
    conf = OmegaConf.create({
        'run_name': run_name,
        'dataset_conf_path': dataset_conf_path,
        'outer_batch_size': 256,
    })

    with JobContext(f"{run_name}_eval"):
        with strategy.scope():
            score_fn = get_scorer_tf_load_model(model_path, max_seq_length=max_seq_len)
            run_rerank_with_conf2(score_fn, conf)


if __name__ == "__main__":
    main()