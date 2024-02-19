import sys
import xmlrpc.client
from omegaconf import OmegaConf

from adhoc.resource.scorer_loader import get_rerank_scorer
from iter_util import load_jsonl
from misc_lib import TELI
from trainer_v2.per_project.transparency.mmp.bias.common import load_car_maker_list
from trainer_v2.per_project.transparency.mmp.bias.inference_w_keyword_swap import run_inference_inner2


def main():
    method = sys.argv[1]
    term_list = load_car_maker_list()
    term_list_set = set(term_list)
    source_path = "output/mmp/bias/car_exp/generic_mention_selected.jsonl"
    score_log_path = f"output/mmp/bias/car_exp/exp3/{method}.tsv"

    l_dict: list[dict] = load_jsonl(source_path)
    l_dict_itr = TELI(l_dict, len(l_dict))
    score_log_f = open(score_log_path, "w")

    scorer = get_rerank_scorer(method)
    score_fn = scorer.score_fn

    run_inference_inner2(score_fn, l_dict_itr, term_list_set, term_list, score_log_f)


if __name__ == "__main__":
    main()
