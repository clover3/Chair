import sys

from omegaconf import OmegaConf

from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank import get_scorer_tf_load_model
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank_w_conf import run_rerank_with_conf_common
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def get_score_fn(conf):
    model_path = conf.model_path
    c_log.info("Building scorer")

    try:
        max_seq_len = conf.max_seq_len
    except KeyError:
        max_seq_len = 256

    score_fn = get_scorer_tf_load_model(model_path, max_seq_length=max_seq_len)
    return score_fn


def main():
    c_log.info(__file__)
    get_scorer_fn = get_score_fn
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    strategy = get_strategy()
    with strategy.scope():
        run_rerank_with_conf_common(conf, get_scorer_fn)


if __name__ == "__main__":
    main()
