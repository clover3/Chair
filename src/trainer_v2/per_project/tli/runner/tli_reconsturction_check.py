import sys

from trainer_v2.custom_loop.run_config2 import get_run_config_for_predict
from trainer_v2.per_project.tli.eval_acc_from_tli import eval_accuracy
from trainer_v2.per_project.tli.runner.inference_all_train_split import get_predictor
from trainer_v2.per_project.tli.token_level_inference import TokenLevelInference

import logging

from trainer_v2.per_project.tli.enum_subseq import enum_subseq_136
from trainer_v2.chair_logging import c_log
from trainer_v2.train_util.arg_flags import flags_parser


def main(args):
    c_log.setLevel(logging.DEBUG)
    run_config = get_run_config_for_predict(args)
    nli_predict_fn = get_predictor(run_config)
    tli_module = TokenLevelInference(nli_predict_fn, enum_subseq_136)
    acc = eval_accuracy(tli_module)
    print("Accuracy: ", acc)



if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
