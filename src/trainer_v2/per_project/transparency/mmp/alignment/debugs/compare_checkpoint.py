import numpy as np
import sys
import tensorflow as tf


def main():
    def normalize(name):
        t =  name.replace("tf_bert_for_sequence_classification/", "")
        t = t.replace("bert_new", "bert")
        return t

    def get_weight_keys(save_path):
        model = tf.keras.models.load_model(save_path, compile=False)
        bert_layer = None
        for layer in model.layers:
            if layer.name == "bert" or layer.name == "bert_new":
                bert_layer = layer
                break

        return {normalize(w.name): w.value() for w in bert_layer.weights}

    keys1 = get_weight_keys(sys.argv[1])
    keys2 = get_weight_keys(sys.argv[2])
    print(keys1.keys())
    print(keys2.keys())

    for key in keys1:
        if key in keys2:
            v1 = keys1[key]
            v2 = keys2[key]
            err = np.sum(np.abs(v1-v2))
            print(key, err)



if __name__ == "__main__":
    main()