import itertools
import sys
import yaml
import tensorflow as tf
from transformers import AutoTokenizer

from table_lib import tsv_iter
from dataset_specific.msmarco.passage.path_helper import get_mmp_train_grouped_sorted_path
from misc_lib import select_third_fourth, TELI
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.eval_helper.eval_line_format import eval_dev_mrr, \
    predict_and_batch_save_scores, score_and_save_score_lines
from trainer_v2.per_project.transparency.mmp.rerank import get_scorer, build_inference_model2
from trainer_v2.per_project.transparency.mmp.trnsfmr_util import get_qd_encoder
from trainer_v2.train_util.arg_flags import flags_parser
from trainer_v2.train_util.get_tpu_strategy import get_strategy
from typing import List, Iterable, Callable, Dict, Tuple, Set
from cpath import output_path, data_path
from misc_lib import path_join




def main():
    quad_tsv_path = get_mmp_train_grouped_sorted_path(0)
    # model_path = config['model_path']
    scores_path = path_join(output_path, "msmarco", "passage", "mmp_train_split_all_scores", "0.scores")
    tuple_itr: Iterable[Tuple[str, str]] = select_third_fourth(tsv_iter(quad_tsv_path))
    batch_size = 1024
    data_size = 1000 * 100
    tuple_itr = itertools.islice(tuple_itr, data_size)
    f = open(scores_path, "w")
    def flush(scores):
        for s in scores:
            f.write("{}\n".format(s))

    strategy = get_strategy(False, "")
    max_seq_length = 256
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def generator():
        for query, document in tuple_itr:
            yield query, document

    def encode(query, document):
        print(query)
        print(document)
        encoded_input = tokenizer.encode_plus(
            (query, document),
            padding="max_length",
            max_length=max_seq_length,
            truncation=True,
            return_tensors="tf"
        )
        return (encoded_input['input_ids'][0], encoded_input['token_type_ids'][0]),

    with strategy.scope():
        # c_log.info("Building scorer")
        # c_log.info("Loading model from %s", model_path)
        # paired_model = tf.keras.models.load_model(model_path, compile=False)
        # inference_model = build_inference_model2(paired_model)

        SpecS = tf.TensorSpec((), dtype=tf.string)
        sig = (SpecS, SpecS)
        ds_qd_pair = tf.data.Dataset.from_generator(
            generator,
            output_signature=sig)

        ds = ds_qd_pair.map(encode, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)

        for batch in iter(ds):
            print(batch)
            break
        c_log.info("Starting predictions")


    c_log.info("Done")


if __name__ == "__main__":
    main()

