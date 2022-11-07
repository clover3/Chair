import tensorflow as tf
import numpy as np

from list_lib import right
from trainer_v2.custom_loop.modeling_common.tf_helper import distribute_dataset
from trainer_v2.train_util.get_tpu_strategy import get_strategy2


def main():
    seq_len = 100
    batch_size = 16
    zeros = np.zeros([seq_len], np.int)
    state = zeros, zeros
    action = zeros
    def build_dataset():
        def raw_data_iter():
            # src_dataset: tf.data.Dataset = self.src_dataset_factory(src_path, False)
            for _ in range(100):
                # items = self.explore(x)
                for _ in range(batch_size):
                    sa = state, action
                    sa_list = [(state, action) for _ in range(10)]
                    reward_hat = 1
                    reward_list = [2 for _ in range(10)]

                    yield {
                        'x': state,
                        'y_hat': action,
                        'y_s_list': right(sa_list),
                        'reward_hat': reward_hat,
                        'reward_list': reward_list,
                    }

        int_list_spec = tf.TensorSpec(shape=(seq_len,), dtype=tf.int32)
        output_signature = {'x': (int_list_spec, int_list_spec),
                            'y_hat': int_list_spec,
                            'y_s_list': tf.TensorSpec(shape=(None, seq_len,), dtype=tf.int32),
                            'reward_hat': tf.TensorSpec(shape=(), dtype=tf.float32),
                            'reward_list': tf.TensorSpec(shape=(None), dtype=tf.float32),
                            }

        return tf.data.Dataset.from_generator(raw_data_iter, output_signature=output_signature)

    dataset = build_dataset()
    dataset = dataset.batch(batch_size)
    strategy = get_strategy2(False, "")
    print(dataset)
    dist_train_dataset = distribute_dataset(strategy, dataset)

    for b in dist_train_dataset:
        for key in b:
            if key == 'x':
                print(key, b[key][0].shape)
            else:
                print(key, b[key].shape)


if __name__ == "__main__":
    main()