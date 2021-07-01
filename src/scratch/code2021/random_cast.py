import tensorflow as tf


def main():
    factor = 0.5

    for _ in range(20):
        r_val = tf.random.uniform([], 0, 1)
        mask = tf.cast(tf.less(r_val, factor), tf.float32)
        print(r_val, mask)



if __name__ == "__main__":
    main()