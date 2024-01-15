import sys
import xmlrpc.client
from omegaconf import OmegaConf
from iter_util import load_jsonl
from misc_lib import TELI
from trainer_v2.per_project.transparency.mmp.bias.common import load_car_maker_list
from trainer_v2.per_project.transparency.mmp.bias.inference_w_keyword_swap import run_inference_inner2


def main():
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)

    term_list = load_car_maker_list()
    term_list_set = set(term_list)
    l_dict: list[dict] = load_jsonl(conf.source_path)
    l_dict_itr = TELI(l_dict, len(l_dict))
    score_log_f = open(conf.score_log_path, "w")
    port = 28122
    server_addr = "localhost"
    proxy = xmlrpc.client.ServerProxy('http://{}:{}'.format(server_addr, port))

    def score_fn(qd_pairs):
        return proxy.predict(qd_pairs)

    run_inference_inner2(score_fn, l_dict_itr, term_list_set, term_list, score_log_f)


if __name__ == "__main__":
    main()
