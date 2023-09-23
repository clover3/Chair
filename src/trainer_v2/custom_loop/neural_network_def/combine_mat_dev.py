from trainer_v2.custom_loop.neural_network_def.combine_mat import cpt_combine_two_way
import tensorflow as tf


def main():
    probs = tf.constant([
        [[0.99, 0.01], [0.5, 0.5]],
    ])
    output = cpt_combine_two_way(probs)
    print(output)


if __name__ == "__main__":
    main()
