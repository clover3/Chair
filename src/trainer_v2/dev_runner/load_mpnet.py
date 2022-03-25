import os

import numpy
import tensorflow as tf
from transformers import TFMPNetModel, MPNetConfig, MPNetTokenizer

from cache import load_from_pickle
from cpath import output_path
from trainer_v2.epr.mpnet import load_tf_mp_net_model


def load_reference_embedding():
    return load_from_pickle("hello_world_sent_transformer")


def main2():
    s_bert_path = os.path.join(output_path, "model", "runs", "paraphrase-mpnet-base-v2",
                                   "tf_model.h5")
    convert_save_path = os.path.join(output_path, "model", "runs", "paraphrase-mpnet-base-v2-keras", "model")

    config_path = os.path.join(output_path, "model", "runs", "paraphrase-mpnet-base-v2",
                               "config.json")
    tokenizer = MPNetTokenizer.from_pretrained("microsoft/mpnet-base")
    configuration = MPNetConfig.from_json_file(config_path)
    model = TFMPNetModel(configuration)
    inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
    print("model.compile")
    model.compile()
    print("model.load_weights(s_bert_path)")
    print("model(inputs)")
    outputs = model(inputs)
    model.load_weights(s_bert_path)
    print("Done")
    hello_world_inptus = tokenizer("hello world ", return_tensors="tf")
    outputs = model(hello_world_inptus)
    verify_output_value(outputs)

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.save(convert_save_path)


def verify_output_value(outputs):
    last_hidden = outputs.last_hidden_state.numpy()
    pooler = outputs.pooler_output.numpy()
    print("last_hidden", last_hidden, last_hidden.shape)
    mean_vector = numpy.mean(last_hidden, axis=1)[0]
    print("mean_vector", mean_vector)
    ref_vector = load_reference_embedding()
    print("ref_vector", ref_vector)
    diff_val = numpy.sum(numpy.abs(mean_vector - ref_vector))
    if diff_val < 1e-3:
        print("Diff Okay")
    else:
        raise ValueError(diff_val)


def load_test():
    convert_save_path = os.path.join(output_path, "model", "runs", "paraphrase-mpnet-base-v2-keras", "model-1")
    tokenizer = MPNetTokenizer.from_pretrained("microsoft/mpnet-base")
    config_path = os.path.join(output_path, "model", "runs", "paraphrase-mpnet-base-v2",
                               "config.json")
    model = load_tf_mp_net_model(config_path, convert_save_path)
    hello_world_inptus = tokenizer("hello world ", return_tensors="tf")
    outputs = model(hello_world_inptus)
    verify_output_value(outputs)
    print("Done")


if __name__ == "__main__":
    load_test()