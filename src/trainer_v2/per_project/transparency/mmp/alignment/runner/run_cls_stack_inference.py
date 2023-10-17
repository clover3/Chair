import os
import pickle
from dataclasses import dataclass

from transformers import AutoTokenizer

from data_generator.tokenizer_wo_tf import get_tokenizer
from table_lib import tsv_iter
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.modeling_common.network_utils import TwoLayerDense
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config
from trainer_v2.per_project.transparency.mmp.alignment.dataset_factory import read_galign_v2, \
    build_galign_from_pair_list
from trainer_v2.per_project.transparency.mmp.alignment.network.cls_stack_pairwise import GAlignClsStackTwoModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import sys
from taskman_client.wrapper3 import report_run3
from trainer_v2.train_util.arg_flags import flags_parser
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2, get_run_config_for_predict


def build_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    hidden_dim = 4 * 768
    classifier_builder = lambda: TwoLayerDense(hidden_dim, 1, 'relu', None)
    return GAlignClsStackTwoModel(tokenizer, classifier_builder)


def save_line_scores(scores, save_path):
    with open(save_path, "w") as f:
        for row in scores:
            s = row[0]
            f.write("{}\n".format(s))


@report_run3
def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config_for_predict(args)
    run_config.print_info()
    max_term_len = 1
    tokenizer = get_tokenizer()

    def build_dataset(input_files, is_for_training):
        itr = tsv_iter(input_files)
        items = [(q, d, int(s)) for q, d, s in itr]
        return build_galign_from_pair_list(
            items, tokenizer, max_term_len)

    strategy = get_strategy_from_config(run_config)
    with strategy.scope():
        eval_dataset = build_dataset(run_config.dataset_config.eval_files_path, False)
        network = build_model()
        network.load_checkpoint(run_config.predict_config.model_save_path)
        outputs = network.get_inference_model().predict(eval_dataset)
        scores = outputs["align_probe"]["align_pred"]
        save_line_scores(scores, run_config.predict_config.predict_save_path)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
