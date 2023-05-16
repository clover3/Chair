import sys

from transformers import AutoTokenizer

from cpath import get_bert_config_path
from taskman_client.task_proxy import get_task_manager_proxy
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config
from trainer_v2.per_project.transparency.mmp.eval_helper.eval_line_format import predict_and_save_scores, \
    eval_dev100_mrr, predict_and_batch_save_scores
from trainer_v2.per_project.transparency.mmp.tt_model.encoders import TermVector
from trainer_v2.per_project.transparency.mmp.tt_model.load_tt_predictor import get_tt10_scorer
from tf_util.lib.tf_funcs import find_layer
from trainer_v2.train_util.arg_flags import flags_parser
from trainer_v2.train_util.get_tpu_strategy import get_strategy
import tensorflow as tf
import numpy as np


def build_layer(paired_model, name, role, dummy_input):
    bert_params = load_bert_config(get_bert_config_path())
    term_encoder = TermVector(bert_params, role)
    orig_layer = find_layer(paired_model, name)
    term_encoder(dummy_input)
    itr = zip(term_encoder.embeddings_layer.weights, orig_layer.embeddings_layer.weights)

    for w1, w2 in itr:
        w1.assign(w2)
    return term_encoder, orig_layer




def main(args):
    model_path = args.output_dir

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    target_q_word = "when"
    target_q_word_id = tokenizer.vocab[target_q_word]
    strategy = get_strategy(args.use_tpu, args.tpu_name)
    term_len = 4
    max_term = 1
    batch_size = 1
    q_input_ids = np.zeros([batch_size, 2, term_len], dtype=int)
    q_input_ids[0, 0, 0] = target_q_word_id
    step = 100
    d_input_ids = get_d_input_ids(100, step)
    alpha = 0.1

    def compare_encoders(term_encoder, orig_layer):
        def print_some_weight(encoder):
            v = encoder.embeddings_layer.weights[0][0][24]
            print(v.numpy())

        print_some_weight(orig_layer)
        print_some_weight(term_encoder)

    paired_model = tf.keras.models.load_model(model_path)
    q_encoder, q_encoder_orig = build_layer(paired_model, "term_vector", "query", q_input_ids)
    d_encoder, d_encoder_orig = build_layer(paired_model, "term_vector_1", "doc", d_input_ids)
    q_reps = q_encoder(q_input_ids)

    cursor = 100
    while cursor < 30000:
        print("Computing from {} to {}".format(cursor, cursor+step))
        d_input_ids = get_d_input_ids(cursor, cursor+step)
        d_reps = d_encoder(d_input_ids)
        d_t = tf.transpose(d_reps, [0, 2, 1])
        qd_term_dot_output = tf.matmul(q_reps, d_t) # [B, M, M]
        d_expanded_tf = tf.nn.gelu(qd_term_dot_output)
        for j in range(2):
            for i in range(step):
                v = d_expanded_tf[0, j, i]
                if v > 1e-2:
                    print(i, j, v)
        cursor += step


def get_d_input_ids(st, ed):
    d_input_ids = np.arange(st, ed, dtype=int)
    d_input_ids = np.expand_dims(d_input_ids, axis=0)
    d_input_ids = np.expand_dims(d_input_ids, axis=2)
    return d_input_ids


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)

