from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank import get_scorer
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank_w_conf import run_rerank_with_conf_common
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def get_score_fn(conf):
    model_path = conf.model_path

    strategy = get_strategy()
    with strategy.scope():
        c_log.info("Building scorer")
        score_fn = get_scorer(model_path)
    return score_fn



def main():
    c_log.info(__file__)
    get_scorer_fn = get_score_fn
    run_rerank_with_conf_common(get_scorer_fn)



if __name__ == "__main__":
    main()
