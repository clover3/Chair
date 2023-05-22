import sys
import tensorflow as tf

from trainer_v2.per_project.transparency.mmp.alignment.debugs.model_load_dev import get_dev_batch
from trainer_v2.per_project.transparency.mmp.probe.align_network import GAlignNetwork
from trainer_v2.per_project.transparency.mmp.probe.probe_network import ProbeOnBERT


def galign_save_load():
    print("galign_save_load")
    base_model_path = sys.argv[1]
    ranking_model = tf.keras.models.load_model(base_model_path, compile=False)
    network = GAlignNetwork(ranking_model)
    temp_model_save_path ="/tmp/model_saved"
    network.model.save(temp_model_save_path)
    loaded_model = tf.keras.models.load_model(temp_model_save_path)
    print(loaded_model)
    batch = get_dev_batch()

    batch['d_term_mask1'] = tf.zeros_like(batch['input_ids1'])
    batch['q_term_mask1'] = tf.zeros_like(batch['input_ids1'])
    batch['d_term_mask2'] = tf.zeros_like(batch['input_ids1'])
    batch['q_term_mask2'] = tf.zeros_like(batch['input_ids1'])
    batch['label1'] = tf.zeros([1, 1])
    batch['label2'] = tf.zeros([1, 1])
    batch['is_valid1'] = tf.zeros([1, 1])
    batch['is_valid2'] = tf.zeros([1, 1])
    output = loaded_model(batch)
    print(output['logits'])
    print("Success")


def probe_save_load():
    print("probe_save_load")
    base_model_path = sys.argv[1]
    ranking_model = tf.keras.models.load_model(base_model_path, compile=False)
    network = ProbeOnBERT(ranking_model)
    temp_model_save_path ="/tmp/probe"
    network.model.save(temp_model_save_path)
    loaded_model = tf.keras.models.load_model(temp_model_save_path)

    batch = get_dev_batch()
    output = loaded_model(batch)
    print(output['logits'])
    print("Done")


if __name__ == "__main__":
    probe_save_load()