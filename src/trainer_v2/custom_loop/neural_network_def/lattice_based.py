import numpy as np
import tensorflow as tf
import tensorflow_lattice as tfl
from tensorflow import keras


class Lattice3(tf.keras.layers.Layer):
    def __init__(self, n_seg, lattice_sizes):
        super(Lattice3, self).__init__()
        self.n_output = 3
        self.n_seg = n_seg
        n_dim = self.n_output * self.n_seg

        def get_lattice_layer(key_input_idx):
            def monotonic_opt(idx):
                if idx % 3 == key_input_idx:
                    return 'increasing'
                else:
                    return 'none'

            monotonicities = list(map(monotonic_opt, range(n_dim)))
            lattice_layer = tfl.layers.Lattice(
                lattice_sizes=lattice_sizes,
                monotonicities=monotonicities,
                output_min=0.0,
                output_max=1.0,
            )
            return lattice_layer

        self.lattice_layers = [get_lattice_layer(i) for i in range(self.n_output)]

    def call(self, inputs, *args, **kwargs):
        flat_inputs = tf.reshape(inputs, [-1, self.n_output * self.n_seg])
        h_list = []
        for i in range(self.n_output):
            h = self.lattice_layers[i](flat_inputs)
            h_list.append(h)
        z = tf.concat(h_list, axis=1)
        y = tf.nn.softmax(z, axis=1)
        return y


def get_lattice(lattice_sizes, monotonicities):
    lattice_layer = tfl.layers.Lattice(
        lattice_sizes=lattice_sizes,
        monotonicities=monotonicities,
        output_min=0.0,
        output_max=1.0,
    )
    return lattice_layer



class MonoCombiner(tf.keras.layers.Layer):
    def __init__(self, width, n_seg):
        super(MonoCombiner, self).__init__()
        self.n_output = 3
        self.n_seg = n_seg
        n_dim = self.n_output * self.n_seg
        lattice_sizes = [width for _ in range(n_dim)]
        self.calibrator = tfl.layers.PWLCalibration(
            units=n_dim,
            input_keypoints=np.linspace(0, 1, num=100),
            dtype=tf.float32,
            output_min=0.0,
            output_max=width - 1.0,
            input_keypoints_type='learned_interior',
            monotonicity='increasing')

        def get_lattice_layer(key_input_idx):
            def monotonic_opt(idx):
                if idx % 3 == key_input_idx:
                    return 'increasing'
                else:
                    return 'none'

            monotonicities = list(map(monotonic_opt, range(n_dim)))
            lattice_layer = get_lattice(lattice_sizes, monotonicities)
            return lattice_layer
        self.lattice_layers = [get_lattice_layer(i) for i in range(self.n_output)]

    def call(self, inputs, *args, **kwargs):
        flat_inputs = tf.reshape(inputs, [-1, self.n_output * self.n_seg])
        h0 = self.calibrator(flat_inputs)

        h_list = []
        for i in range(self.n_output):
            h = self.lattice_layers[i](h0)
            h_list.append(h)
        z = tf.concat(h_list, axis=1)
        y = tf.nn.softmax(z, axis=1)
        return y


def get_average(t, input_mask, axis):
    t2 = tf.reduce_sum(t, axis)
    eps = 1e-6
    n_valid = tf.reduce_sum(tf.cast(input_mask, tf.float32), axis=1, keepdims=True) + eps
    return tf.divide(t2, n_valid)


class MonoSortCombiner(tf.keras.layers.Layer):
    def __init__(self):
        super(MonoSortCombiner, self).__init__()
        self.inner_combiner = MonoCombiner(3, 3)

    def call(self, local_decisions, input_mask, *args, **kwargs):
        max_l_probs = tf.reduce_max(local_decisions, axis=1)  # [batch_size, 3]
        min_l_probs = tf.reduce_min(local_decisions, axis=1)  # [batch_size, 3]
        avg_l_probs = get_average(local_decisions, input_mask, 1)  # [batch_size, 3]
        rep = tf.concat([max_l_probs, min_l_probs, avg_l_probs], axis=1)
        return self.inner_combiner(rep)



def main():
    local_d = [
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1]
    ]
    label = 1
    model_input = keras.layers.Input(shape=[2, 3])
    lattice_inputs_tensor = tf.reshape(model_input, [-1, 6])
    y = Lattice3()(lattice_inputs_tensor)
    model = keras.models.Model(inputs=[model_input], outputs=[y])
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=keras.optimizers.Adagrad(learning_rate=1.0))

    model.fit([local_d], [label], epochs=30)
    y_pred = model.predict([local_d])
    print(y_pred)


if __name__ == "__main__":
    main()