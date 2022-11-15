import os
import sys
import tensorflow as tf

from cache import save_to_pickle
from data_generator.bert_input_splitter import split_p_h_with_input_ids
from data_generator.tokenizer_wo_tf import get_tokenizer, ids_to_text
from trainer_v2.custom_loop.dataset_factories import get_classification_dataset
from trainer_v2.custom_loop.definitions import ModelConfig600_3
from trainer_v2.custom_loop.per_task.nli_ts_util import load_local_decision_model
from trainer_v2.custom_loop.run_config2 import get_eval_run_config2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from taskman_client.wrapper3 import report_run3
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    run_config = get_eval_run_config2(args)

    input_files = args.eval_input_files
    model_config = ModelConfig600_3()
    eval_dataset = get_classification_dataset(input_files, run_config, model_config, False)
    model = load_local_decision_model(run_config.eval_config.model_save_path)

    maybe_step = 1000
    res = model.predict(eval_dataset, steps=maybe_step, verbose=True)
    l_decision, g_decision = res
    n_item = len(l_decision)
    save_to_pickle(res, "double_negation_analysis")
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

    def parse_input_id(input_ids):
        input_ids = input_ids.numpy().tolist()
        seg1 = input_ids[:300]
        p1, h1 = split_p_h_with_input_ids(seg1, seg1)
        seg2 = input_ids[300:]
        p2, h2 = split_p_h_with_input_ids(seg2, seg2)
        return p1, h1, h2

    def double_negation_entailment(l_pred, label, g_pred):
        return l_pred[0] == 2 and l_pred[1] == 2 and label == 0

    def condition_contradiction_mismatch(l_pred, label, g_pred):
        return l_pred[0] == 1 and l_pred[1] == 2 and label == 0


    tokenizer = get_tokenizer()
    for i in range(n_item):
        l_decision_item = l_decision[i]
        label = labels[i].numpy().tolist()
        l_pred = tf.argmax(l_decision_item, axis=1).numpy().tolist()
        g_pred = tf.argmax(g_decision[0][i]).numpy().tolist()
        input_ids = input_ids_list[i]

        if double_negation_entailment(l_pred, label, g_pred):
            # print("Maybe double negation at ", i)
            print()
            if g_pred == label:
                print("This is unexpected correct {} = {}".format(g_pred, label))
            print("Not matching contradiction ", i)
            p, h1, h2 = parse_input_id(input_ids)
            print("Prem:", ids_to_text(tokenizer, p))
            print("H1:", ids_to_text(tokenizer, h1))
            print("H2:", ids_to_text(tokenizer, h2))


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
