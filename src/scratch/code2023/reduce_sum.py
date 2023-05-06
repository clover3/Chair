import tensorflow as tf


def main():
    tensor_d = {
        "tensor": tf.constant([10]),
        "tensor2": tf.constant([10])
    }
    loss = tf.reduce_sum(list(tensor_d.values()))
    print(loss)


if __name__ == "__main__":
    main()
