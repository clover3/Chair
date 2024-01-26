import sys

from cpath import output_path
from iter_util import load_jsonl
from misc_lib import path_join
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.bias.common import load_car_maker_list
from trainer_v2.per_project.transparency.mmp.bias.inference_w_keyword_swap import run_inference_inner2
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank import get_scorer_tf_load_model
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def main():
    model_path = sys.argv[1]
    source_path = sys.argv[2]

    term_list = load_car_maker_list()
    term_list_set = set(term_list)

    l_dict: list[dict] = load_jsonl(source_path)
    score_log_f = open(path_join(output_path, "mmp", "bias", "car_exp", "exp3", f"res.tsv"), "w")
    strategy = get_strategy()
    batch_size = 256
    with strategy.scope():
        c_log.info("Building scorer")
        score_fn = get_scorer_tf_load_model(model_path, batch_size)

    run_inference_inner2(score_fn, l_dict, term_list_set, term_list, score_log_f)


if __name__ == "__main__":
    main()
