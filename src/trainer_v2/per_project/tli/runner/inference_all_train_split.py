import itertools
import pickle
import sys
from typing import List, Iterable, Callable, Dict, Tuple, Set
from cache import save_to_pickle
from data_generator.tokenizer_wo_tf import get_tokenizer
from trainer_v2.custom_loop.per_task.nli_ts_helper import get_local_decision_nlits_core
from trainer_v2.custom_loop.run_config2 import get_run_config_for_predict
from trainer_v2.per_project.tli.token_level_inference import TokenLevelInference, nc_max_e_avg_reduce_then_softmax
from cpath import output_path
from misc_lib import path_join

import numpy as np
from trainer_v2.per_project.tli.enum_subseq import enum_subseq_136
from dataset_specific.mnli.mnli_reader import MNLIReader
from trainer_v2.chair_logging import c_log
from trainer_v2.train_util.arg_flags import flags_parser

NLIPredictorSig = Callable[[List[Tuple[str, str]]], List[List[float]]]


def get_predictor(run_config) -> NLIPredictorSig:
    encoder_name = "concat"
    tokenizer = get_tokenizer()
    nlits = get_local_decision_nlits_core(run_config, encoder_name)

    def predict_fn(pair_list: List[Tuple[str, str]]) -> List[List[float]]:
        t_list = []
        for p, h in pair_list:
            p_tokens = tokenizer.tokenize(p)
            h_tokens = tokenizer.tokenize(h)
            t = nlits.encode_fn(p_tokens, h_tokens, h_tokens)
            t_list.append(t)
        l_decision_list, g_decision_list = nlits.predict(t_list)
        second_l_decision = [d[1] for d in l_decision_list]
        return second_l_decision
    return predict_fn


def main(args):
    run_config = get_run_config_for_predict(args)
    nli_predict_fn: NLIPredictorSig = get_predictor(run_config)
    job_id = int(run_config.dataset_config.eval_files_path)
    tli_module = TokenLevelInference(nli_predict_fn, enum_subseq_136)

    reader = MNLIReader()
    split = "train"
    c_log.info("Job %d", job_id)
    job_size = 1024
    n_job = reader.get_data_size(split) / job_size
    st = job_size * job_id
    ed = st + job_size
    itr = itertools.islice(reader.load_split(split), st, ed)
    tli_payload = []
    for pair in itr:
        tli_payload.append((pair.premise, pair.hypothesis))

    if not tli_payload:
        print("tli_payload is empty")

    tli_output_list = tli_module.do_batch(tli_payload)
    tli_dict: Dict[Tuple[str, str], np.array] = dict(zip(tli_payload, tli_output_list))
    save_path = path_join(output_path, "tli", "nli_train_pred", str(job_id))
    pickle.dump(tli_dict, open(save_path, "wb"))


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
