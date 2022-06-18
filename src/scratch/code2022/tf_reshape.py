import tensorflow as tf


def main():
    batch_size = 3
    seq_len = 4
    arr1 = tf.ones([batch_size, seq_len])
    arr2 = tf.ones([batch_size, seq_len]) * 2
    print('arr1', arr1)
    print('arr2', arr2)
    num_window = 2
    concated = tf.concat([arr1, arr2], axis=0)
    output = tf.reshape(concated, [batch_size, num_window, seq_len])
    maybe1 = output[:, 0]
    maybe2 = output[:, 1]
    print('maybe1', maybe1)
    print('maybe2', maybe2)



    return NotImplemented


if __name__ == "__main__":
    main()