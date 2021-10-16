import sys

import tensorflow as tf


def main():
    target_name = "bert/encoder/layer_6/intermediate/dense/bias"
    target_name = "bert/embeddings/LayerNorm/gamma"
    v = tf.train.load_variable(sys.argv[1], target_name)
    print(list(v))




if __name__ == "__main__":
    main()