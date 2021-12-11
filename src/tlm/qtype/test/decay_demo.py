import tensorflow as tf

from tlm.qtype.qtype_embeddings import decay_fn


def main():
    init_temperature = tf.constant(1.0)
    rate = 0.1
    decay_steps = 100000
    h = tf.constant([[0.1, 0.2, 0.8, -0.2]])
    steps = [1000, 5000, ]
    steps.extend(range(10000, 1000000, 10000))
    for current_step in steps:
        temperature = decay_fn(init_temperature, rate, float(current_step), decay_steps)
        output = tf.nn.softmax(h / temperature)
        print(current_step, output)



if __name__ == "__main__":
    main()