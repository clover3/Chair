import h5py
from tensorflow import keras

from trainer_v2.chair_logging import c_log


def load_weights_from_hdf5(model, h5_path, map_to_stock_fn, n_expected_restore):
    param_storage = h5py.File(h5_path, 'r')
    prefix = "bert"
    loaded_weights = set()
    skip_count = 0
    weight_value_tuples = []
    skipped_weight_value_tuples = []
    bert_params = model.weights
    param_values = keras.backend.batch_get_value(model.weights)
    for ndx, (param_value, param) in enumerate(zip(param_values, bert_params)):
        stock_name = map_to_stock_fn(param.name, prefix)
        if stock_name in param_storage:
            ckpt_value = param_storage[stock_name]
            if param_value.shape != ckpt_value.shape:
                c_log.warn("loader: Skipping weight:[{}] as the weight shape:[{}] is not compatible "
                           "with the checkpoint:[{}] shape:{}".format(param.name, param.shape,
                                                                      stock_name, ckpt_value.shape))
                skipped_weight_value_tuples.append((param, ckpt_value))
                continue

            weight_value_tuples.append((param, ckpt_value))
            loaded_weights.add(stock_name)
        else:
            c_log.info("loader: No value for:[{}]".format(param.name))
            skip_count += 1

    keras.backend.batch_set_value(weight_value_tuples)
    if n_expected_restore is not None:
        if n_expected_restore == len(weight_value_tuples):
            pass
        else:
            msg = "Done loading {} BERT weights from checkpoint into {} (prefix:{}). " \
                       "Count of weights not found in the checkpoint was: [{}]. " \
                       "Count of weights with mismatched shape: [{}]".format(
                len(weight_value_tuples), model, prefix, skip_count, len(skipped_weight_value_tuples))
            c_log.warning(msg)
            c_log.warning("Expected to restore %d but actually restored %d ", n_expected_restore, len(weight_value_tuples))

            param_storage_keys = set(param_storage.keys())
            c_log.warning("Unused weights from checkpoint: %s",
                       "\n\t" + "\n\t".join(sorted(param_storage_keys.difference(loaded_weights))))
            raise ValueError("Checkpoint load exception")
    else:
        c_log.info("Done loading {} weights from checkpoint".format(len(weight_value_tuples)))
    return skipped_weight_value_tuples