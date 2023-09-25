import os
import sys
import tensorflow as tf
from tensorflow.python.distribute.collective_all_reduce_strategy import CollectiveAllReduceStrategy

from cache import save_to_pickle
from data_generator.bert_input_splitter import split_p_h_with_input_ids
from data_generator.tokenizer_wo_tf import get_tokenizer, ids_to_text
from misc_lib import ceil_divide
from trainer_v2.custom_loop.dataset_factories import get_classification_dataset
from trainer_v2.custom_loop.definitions import ModelConfig600_3
from trainer_v2.custom_loop.modeling_common.tf_helper import distribute_dataset
from trainer_v2.custom_loop.per_task.nli_ts_util import load_local_decision_model_n_label_3
from trainer_v2.custom_loop.run_config2 import get_eval_run_config2
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from taskman_client.wrapper3 import report_run3
from trainer_v2.train_util.arg_flags import flags_parser


def extract_x_y(eval_dataset, n_item):
    labels = []
    input_ids_list = []
    for t in iter(eval_dataset):
        x, y = t

        for y_i in y:
            labels.append(y_i)
        input_ids, seg_ids = x
        for input_ids_one in input_ids:
            input_ids_list.append(input_ids_one)

        if len(labels) > n_item:
            break
    return input_ids_list, labels


def run_print_failure(run_config, save_name, is_target_failure):
    input_files = args.eval_input_files
    model_config = ModelConfig600_3()
    num_item = 10000
    run_config.common_run_config.batch_size = 64
    maybe_step = ceil_divide(num_item, run_config.common_run_config.batch_size)

    strategy = get_strategy_from_config(run_config)
    with strategy.scope():
        model = load_local_decision_model_n_label_3(run_config.eval_config.model_save_path)
        eval_dataset = get_classification_dataset(input_files, run_config, model_config, False)
        dist_dataset = distribute_dataset(strategy, eval_dataset)
        res = model.predict(dist_dataset, steps=maybe_step, verbose=True)
    l_decision, g_decision = res
    n_item = len(l_decision)
    save_to_pickle(res, save_name)
    input_ids_list, labels = extract_x_y(eval_dataset, n_item)

    def parse_input_id(input_ids):
        input_ids = input_ids.numpy().tolist()
        seg1 = input_ids[:300]
        p1, h1 = split_p_h_with_input_ids(seg1, seg1)
        seg2 = input_ids[300:]
        p2, h2 = split_p_h_with_input_ids(seg2, seg2)
        return p1, h1, h2

    tokenizer = get_tokenizer()
    for i in range(n_item):
        l_decision_item = l_decision[i]
        label = labels[i].numpy().tolist()
        l_pred = tf.argmax(l_decision_item, axis=1).numpy().tolist()
        g_pred = tf.argmax(g_decision[0][i]).numpy().tolist()
        input_ids = input_ids_list[i]

        if is_target_failure(l_pred, label, g_pred):
            # print("Maybe double negation at ", i)
            print()
            if g_pred == label:
                print("This is unexpectedly correct {} = {}".format(g_pred, label))
            print("Data ID ", i)
            p, h1, h2 = parse_input_id(input_ids)
            print("Prem:", ids_to_text(tokenizer, p))
            print("H1:", ids_to_text(tokenizer, h1))
            print("H2:", ids_to_text(tokenizer, h2))


@report_run3
def main(args):
    run_config = get_eval_run_config2(args)

    def double_negation_entailment(l_pred, label, g_pred):
        return l_pred[0] == 2 and l_pred[1] == 2 and label == 0

    def condition_contradiction_mismatch(l_pred, label, g_pred):
        return (l_pred[0] == 1 and l_pred[1] == 2 and label == 1) or (l_pred[0] == 2 and l_pred[1] == 1 and label == 1)

    save_name = "double_negation_analysis"
    save_name = "neutral_contradiction_neutral"
    print(save_name)
    run_print_failure(run_config, save_name, condition_contradiction_mismatch)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
