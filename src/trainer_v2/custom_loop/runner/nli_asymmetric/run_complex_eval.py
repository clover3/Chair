import os
import sys
from collections import defaultdict, Counter

from tab_print import print_table

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from misc_lib import SuccessCounter, two_digit_float
from trainer_v2.custom_loop.modeling_common.tf_helper import distribute_dataset
from trainer_v2.custom_loop.per_task.nli_ts_util import load_local_decision_model_n_label_3
from trainer_v2.custom_loop.per_task.ts_util import get_local_decision_layer_from_model_by_shape
from taskman_client.wrapper3 import report_run3
from trainer_v2.custom_loop.dataset_factories import get_classification_dataset
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2_nli
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config
from trainer_v2.train_util.arg_flags import flags_parser
import numpy as np

class ModelConfig(ModelConfigType):
    max_seq_length = 600
    num_classes = 3


@report_run3
def main(args):
    run_config: RunConfig2 = get_run_config2_nli(args)
    strategy = get_strategy_from_config(run_config)

    with strategy.scope():
        model_path = run_config.eval_config.model_save_path
        predictor = load_local_decision_model_n_label_3(model_path)

    model_config = ModelConfig()

    def dataset_factory(input_files, is_for_training):
        return get_classification_dataset(input_files, run_config, model_config, is_for_training)
    eval_dataset = dataset_factory(run_config.dataset_config.eval_files_path, False)
    # x_dataset = eval_dataset.map(lambda x, y: ((x[0], x[1], x[2], x[3]),))
    dataset = distribute_dataset(strategy, eval_dataset)
    l_decision, g_decision_l = predictor.predict(dataset, steps=run_config.eval_config.eval_step)
    g_decision = g_decision_l[0]
    print('l_decision', l_decision.shape)
    assert len(g_decision_l) == 1
    print('g_decision', g_decision.shape)
    data_size = run_config.common_run_config.batch_size * run_config.eval_config.eval_step
    print("Maybe data_size: ", data_size)
    dataset = eval_dataset.unbatch()
    iterator = iter(dataset)

    module_list = ["local_1", "local_2", "global"]
    sc_d = defaultdict(SuccessCounter)
    confusion_d = defaultdict(Counter)
    for i in range(data_size):
        item = next(iterator)
        x, y = item
        l_pred = np.argmax(l_decision[i], axis=1)
        g_pred = np.argmax(g_decision[i], axis=0)
        y_v = y.numpy().tolist()
        assert type(y_v) == int
        d = {
            "local_1": l_pred[0],
            "local_2": l_pred[1],
            "global": g_pred,
        }

        for key in module_list:
            pred = d[key]
            sc_d[key + "_acc"].add(y_v == pred)
            sc_d[key + "_e_rate"].add(pred == 0)
            sc_d[key + "_n_rate"].add(pred == 1)
            sc_d[key + "_c_rate"].add(pred == 2)
            confusion_d[key][(y_v, pred)] += 1



    metrics = ["acc", "e_rate", "n_rate", "c_rate"]

    head = ["module"] + metrics
    table = [head]
    for key in module_list:
        row = ["{0}".format(key)]
        for m in metrics:
            row.append("{0:.2f}".format(sc_d["{}_{}".format(key, m)].get_suc_prob()))
        table.append(row)
    print_table(table)

    for key in module_list:
        head = [key, "e", "n", "c", "pred"]
        table = [head]
        cf = confusion_d[key]
        n_total = sum(cf.values())
        for y in range(3):
            row = ["enc"[y]]
            for pred in range(3):
                portion = cf[y, pred] / n_total
                row.append(two_digit_float(portion))
            table.append(row)
        table.append(["gold"])
        print_table(table)
        print()

if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


