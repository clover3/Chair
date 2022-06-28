import tensorflow as tf

from trainer_v2.custom_loop.neural_network_def.role_aug import DummyZeroLayer


def main():
    lower_out = tf.ones([2, 16, 728])

    dzl = DummyZeroLayer(32)
    layer_output = dzl(lower_out)

    print(layer_output)


if __name__ == "__main__":
    main()