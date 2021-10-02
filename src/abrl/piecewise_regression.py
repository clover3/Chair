import numpy as np
import tensorflow as tf

from list_lib import lmap
from misc_lib import two_digit_float


def neg(x):
    return -x


def static_calculation():
    X, Y = load_data('float_18')
    X, Y = load_data('id_list_68')
    get_piecewise_regression(X, Y)


def load_data(name):
    data_d = {
        'float_18':        [
        (0.001,	-17.15),
        (0.002,	-9.12),
        (0.005,	-9.64),
        (0.01,	-6.13),
        (0.02,	-4.75),
        (0.05,	-0.01),
        (0.1,	1.01),
        (0.2,	1.10),
        (0.5,	1.11),
        (1,	    1.11)
    ],
        'id_list_68':[
                   (0.001,	 -17.43),
                   (0.002,	 -10.53),
                   (0.005,	 -3.92),
                   (0.01,	 -0.21),
                   (0.02,	 1.26),
                   (0.05,	 5.14),
                   (0.1,	 5.82),
                   (0.2,	 6.33),
                   (0.5,	 6.69),
                   (1,	     6.73),
        ]
    }

    data_rev = data_d[name][::-1]
    raw_X, raw_Y = zip(*data_rev)
    # X = lmap(neg, raw_X)
    # Y = lmap(neg, raw_Y)
    return raw_X, raw_Y


def get_piecewise_regression(X, Y):
    X = lmap(neg, X)
    Y = lmap(neg, Y)

    b0 = Y[0]
    n = len(X)
    nan = float("nan")
    W = [nan] * n
    for k in range(1, n):
        W[k] = (Y[k] - Y[k-1])/(X[k] - X[k-1]) - sum(W[1:k])

    approx_Y = [nan] * n

    approx_Y[0] = b0
    for k in range(1, n):
        approx_Y[k] = approx_Y[k-1] + (X[k] - X[k-1]) * sum(W[1:k+1])

    print("X", X)
    print("Y", Y)
    print("W:", W)
    print("b0", b0)
    print("approx_Y", approx_Y)


class RegressionLayerOld(tf.keras.layers.Layer):
    def __init__(self):
        super(RegressionLayerOld, self).__init__()

        pivot_X_np = np.array([1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001])
        self.num_point = len(pivot_X_np)

        answer_w = np.array([0.0, 0.03333333333333337, 0.8666666666666675, 19.499999999999996, 137.6,
                             -20.0, 564.0000000000001, -875.3333333333339, 8203.333333333332])
        answer_w = np.array([0.0, 0.03333333333333337, 0.8666666666666675, 19.499999999999996, 137.6,
                             -20.0, 564.0000000000001, -875.3333333333339, 8203.333333333332, 0.0])

        answer_b = 1.11
        init_w = np.zeros([self.num_point], np.float32)
        init_b =0.0
        self.W = tf.Variable(initial_value=init_w,  trainable=True, dtype=tf.float32)
        self.b = tf.Variable(initial_value=init_b, trainable=True)
        self.pivot_X = tf.Variable(initial_value=pivot_X_np, dtype=tf.float32,
                                   trainable=False)

    def call(self, input_X, **kwargs):
        # W_abs = tf.abs(self.W)
        W_abs = tf.math.exp(self.W)
        pivot_X = self.pivot_X
        y0 = self.b
        y_list = [y0]
        for j in range(1, self.num_point):
            s_list = []
            for k in range(0, j):
                s_i = W_abs[k] * (pivot_X[j] - pivot_X[k])
                s_list.append(s_i)

            s = tf.reduce_sum(s_list)
            y_j = self.b + s
            y_list.append(y_j)
        Y = tf.stack(y_list)
        is_less = tf.less_equal(input_X, tf.expand_dims(pivot_X, 0))
        is_last_less_non_end = tf.logical_and(is_less[:, :-1],
                                              tf.logical_not(is_less[:, 1:]))
        is_last_less_end = tf.reduce_all(tf.logical_not(is_last_less_non_end),
                                         axis=1,
                                         keepdims=True)
        is_last_less = tf.concat([is_last_less_non_end, is_last_less_end], axis=1)
        select_mask = tf.cast(is_last_less, tf.float32)
        nearest_Y = tf.reduce_sum(Y * select_mask, axis=1)
        return nearest_Y

gold_pivot_X_np = np.array([1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001])
answer_w = np.array([0.0, 0.03333333333333337, 0.8666666666666675, 19.499999999999996, 137.6,
                     -20.0, 564.0000000000001, -875.3333333333339, 8203.333333333332])
answer_b = 1.11


class RegressionLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(RegressionLayer, self).__init__(name=name)

        arr = []
        k = 1
        for _ in range(14):
            arr.append(k)
            k = k / 2
        pivot_X_np = np.array(arr)
        self.num_point = len(pivot_X_np)

        init_w = np.zeros([self.num_point], np.float32)
        init_b = 0.0
        self.W = tf.Variable(initial_value=init_w,  trainable=True, dtype=tf.float32)
        self.b = tf.Variable(initial_value=init_b, trainable=True)
        self.pivot_X = tf.Variable(initial_value=pivot_X_np, dtype=tf.float32,
                                   trainable=False)
        self.params = [self.W, self.b, self.pivot_X]

    def call(self, input_X, **kwargs):
        # W_abs = tf.abs(self.W)
        W_abs = tf.math.exp(self.W)
        pivot_X = self.pivot_X
        pivot_X_ex = tf.expand_dims(pivot_X, 0)
        # [Batch, num_pivot]
        diff = pivot_X_ex - input_X
        pos_diff = tf.math.maximum(diff, 0)
        prod = -pos_diff * W_abs
        pred_y = tf.reduce_sum(prod, axis=1) + self.b
        return pred_y


class RegressionLayer2(tf.keras.layers.Layer):
    def __init__(self):
        super(RegressionLayer2, self).__init__()

        arr = []
        k = 1
        n_point = 14
        for _ in range(n_point):
            arr.append(k)
            k = k / 2
        pivot_X_np = np.array(arr)
        print(arr)
        self.num_point = len(pivot_X_np)
        self.k = tf.Variable(initial_value=1.4,  trainable=True, dtype=tf.float32)
        self.b = tf.Variable(initial_value=6.0, trainable=True)
        self.b2 = tf.Variable(initial_value=-4.0, trainable=True)
        self.pivot_X = tf.Variable(initial_value=pivot_X_np, dtype=tf.float32,
                                   trainable=False)
        print(np.log(pivot_X_np))
        scale = np.array(list(range(1, n_point+1)))
        self.W = scale * self.k + self.b2
        self.params = [self.k, self.b, self.W, self.pivot_X]

    def call(self, input_X, **kwargs):
        # W_abs = tf.abs(self.W)
        W_abs = tf.math.exp(self.W)
        pivot_X = self.pivot_X
        pivot_X_ex = tf.expand_dims(pivot_X, 0)
        # [Batch, num_pivot]
        diff = pivot_X_ex - input_X
        pos_diff = tf.math.maximum(diff, 0)
        prod = -pos_diff * W_abs
        pred_y = tf.reduce_sum(prod, axis=1) + self.b
        return pred_y


class LogRegression(tf.keras.layers.Layer):
    def __init__(self):
        super(LogRegression, self).__init__()

        self.k = tf.Variable(initial_value=0.01, trainable=True)
        self.a = tf.Variable(initial_value=1.0, trainable=True)
        self.b = tf.Variable(initial_value=1.0, trainable=True)
        self.params = [self.k, self.a, self.b]

    def call(self, input_X, **kwargs):
        v = -self.a * tf.math.log(self.k/input_X) + self.b
        return tf.reshape(v, [-1])


def get_piecewise_regression_model_old():
    input_X = tf.keras.Input(shape=[1], name="X", dtype=tf.float32)
    layer = RegressionLayerOld()
    pred_y = layer(input_X)
    return tf.keras.Model(inputs=[input_X], outputs=[pred_y])


def get_piecewise_regression_model(model_class):
    input_X = tf.keras.Input(shape=[1], name="X", dtype=tf.float32)
    layer = model_class()
    pred_y = layer(input_X)
    return tf.keras.Model(inputs=[input_X], outputs=[pred_y])


def get_log_regression_model():
    input_X = tf.keras.Input(shape=[1], name="X", dtype=tf.float32)
    layer = LogRegression()
    pred_y = layer(input_X)
    return tf.keras.Model(inputs=[input_X], outputs=[pred_y])


def main_tf():
    # X, Y = load_data('float_18')
    X, Y = load_data('id_list_68')

    model = get_piecewise_regression_model(RegressionLayer)
    # optimizer = tf.keras.optimizers.Adam(1e-1)
    optimizer = tf.keras.optimizers.Adam(1e-2)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.MSE)
    sub_model = tf.keras.Model(model.input, outputs=[model.layers[1].output])
    print("model params:")
    for param in sub_model.layers[1].params:
        print(param.numpy())


    def show_prediction():
        prediction = model.predict(X)
        error = prediction - Y
        print("answer", Y)
        # print("sub_model: ", sub_model.predict(X))
        print("prediction: ", lmap(two_digit_float, prediction))
        print("Error", lmap(two_digit_float, error))

    show_prediction()
    model.fit(X, Y, epochs=1000, verbose=0)
    print("eval", model.evaluate(X, Y))
    show_prediction()
    print("model params:")
    for param in sub_model.layers[1].params:
        print(param.numpy())



if __name__ == "__main__":
    main_tf()