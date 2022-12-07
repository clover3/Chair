import sys

from cache import save_list_to_jsonl
from cpath import get_bert_config_path
from trainer.promise import PromiseKeeper, list_future
from trainer_v2.custom_loop.definitions import ModelConfig300_2
from trainer_v2.custom_loop.inference import InferenceHelper
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2, get_run_config_for_predict
from trainer_v2.custom_loop.train_loop import tf_run, load_model_by_dir_or_abs
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config
from trainer_v2.per_project.cip.precomputed_cip import get_cip_pred_splits_iter
from trainer_v2.per_project.cip.tfrecord_gen import SelectAll, encode_separate_core
from trainer_v2.train_util.arg_flags import flags_parser
import tensorflow as tf
from typing import List, Iterable, Callable, Dict, Tuple, Set


class CIPModule:
    def __init__(self):
        pass

    def predict(self, a,):
        pass


# Show how CIP predictor predicts
def main(args):
    run_config: RunConfig2 = get_run_config_for_predict(args)
    run_config.print_info()
    strategy = get_strategy_from_config(run_config)
    model_save_path = run_config.predict_config.model_save_path

    model_factory = lambda: load_model_by_dir_or_abs(model_save_path)
    def dataset_factory(payload: List):
        def generator():
            yield from payload

        int_list = tf.TensorSpec(shape=(model_config.max_seq_length,), dtype=tf.int32)
        output_signature = (int_list, int_list)
        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        return dataset

    predictor = InferenceHelper(model_factory, dataset_factory, strategy)
    predictor.model.summary()
    model_config = ModelConfig300_2()
    split, itr, _ = get_cip_pred_splits_iter()[0]

    pk = PromiseKeeper(predictor.predict)
    future_lists = []
    for comparison in itr:
        futures = []
        for st, ed in comparison.ts_input_info_list:
            triplet = encode_separate_core(comparison.hypo, 300, st, ed)
            input_ids, input_mask, segment_ids = triplet
            e = (input_ids, segment_ids)
            futures.append(pk.get_future(e))

        future_lists.append(futures)


    pk.do_duty(True)
    outputs = []
    for f in future_lists:
        per_comp = []
        for e in list_future(f):
            per_comp.append(e.tolist())
        outputs.append(per_comp)
    save_list_to_jsonl(outputs, run_config.predict_config.predict_save_path)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


