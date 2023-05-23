from trainer_v2.per_project.tli.eval_acc_from_tli import eval_accuracy
from trainer_v2.per_project.tli.token_level_inference import TokenLevelInference

import logging

from trainer_v2.per_project.tli.enum_subseq import enum_subseq_136
from trainer_v2.chair_logging import c_log
from trainer_v2.keras_server.name_short_cuts import get_pep_cache_client


def main():
    c_log.setLevel(logging.DEBUG)
    nli_predict_fn = get_pep_cache_client()
    tli_module = TokenLevelInference(nli_predict_fn, enum_subseq_136)
    acc = eval_accuracy(tli_module)
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main()