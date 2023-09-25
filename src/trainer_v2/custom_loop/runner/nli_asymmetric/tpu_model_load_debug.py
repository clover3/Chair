import os

import tensorflow as tf

from cpath import common_model_dir_root
from taskman_client.task_proxy import get_local_machine_name
from trainer_v2.custom_loop.modeling_common.tf_helper import distribute_dataset
from trainer_v2.custom_loop.per_task.nli_ts_util import load_local_decision_model_n_label_3
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def main():
    machine_name = get_local_machine_name()
    is_tpu = machine_name not in ["GOSFORD", "ingham.cs.umass.edu"]

    if is_tpu:
        model_path = "gs://clovertpu/training/model/nli_ts_run34_0/model_25000"
        strategy = get_strategy(True, "local")
    else:
        model_path = os.path.join(common_model_dir_root, 'runs', "nli_ts_run34_0", "model_25000")
        strategy = get_strategy(False, "")

    with strategy.scope():
        model = load_local_decision_model_n_label_3(model_path)

    batch_size = 16
    num_items = 10
    dummy_p = [0] * 200
    input_list = []
    for i in range(num_items):
        x = dummy_p, dummy_p, dummy_p, dummy_p
        input_list.append(tuple(x))
    while len(input_list) % batch_size:
        input_list.append(input_list[-1])

    dataset = tf.data.Dataset.from_tensor_slices(input_list)

    def reform(row):
        x = row[0], row[1], row[2], row[3]
        return x,
    dataset = dataset.map(reform)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = distribute_dataset(strategy, dataset)
    l_decision, g_decision = model.predict(dataset, steps=1)
    print(l_decision)
    print("Maybe success")


if __name__ == "__main__":
    main()