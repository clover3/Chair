import tensorflow as tf

from abrl.piecewise_regression import RegressionLayer


def get_shape_list2(tensor):
    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(input=tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


class ConvexLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim1=100, hidden_dim2=100):
        super(ConvexLayer, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_dim1, activation='relu')
        self.dense2 = tf.keras.layers.Dense(hidden_dim2, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, inputs, **kwargs):
        # [batch, num_feature, 3]
        shape = get_shape_list2(inputs)
        random_v = tf.abs(tf.random.normal(stddev=0.1, shape=shape))

        def all_network(x):
            h = self.dense1(x)
            h2 = self.dense2(h)
            h3 = self.dense3(h2)
            return h3
        h_out = all_network(inputs)
        h_higher = all_network(inputs + random_v)
        h_lower = all_network(inputs - random_v)

        h_diff1 = h_higher - h_out # This should be positive
        h_diff2 = h_out - h_lower # This should be positive
        increasing_loss1 = tf.reduce_mean(tf.math.maximum(-h_diff1, 0))
        increasing_loss2 = tf.reduce_mean(tf.math.maximum(-h_diff2, 0))
        self.add_loss(increasing_loss1)
        self.add_loss(increasing_loss2)
        return h_out


class ConvexLayer2(tf.keras.layers.Layer):
    def __init__(self, hidden_dim1=100, hidden_dim2=100, num_features=7):
        super(ConvexLayer2, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_dim1, activation='relu')
        self.dense2 = tf.keras.layers.Dense(hidden_dim2, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1)
        self.k = tf.Variable(tf.random_normal_initializer(0.01)([num_features]), trainable=True)

    def call(self, inputs, **kwargs):
        is_used = tf.less(-0.5, inputs)
        shape = get_shape_list2(inputs)
        random_v = tf.abs(tf.random.normal(stddev=0.1, shape=shape))
        random_v = random_v * tf.cast(is_used, tf.float32)

        def all_network(x_input):
            x_input_norm_inv = 1 / x_input # This is larger than 1
            # When k = input, feature_value = 0
            feature_value = -tf.math.log(tf.multiply(self.k, x_input_norm_inv))
            feature_value_masked = tf.where(tf.math.is_nan(feature_value),
                                            tf.zeros_like(feature_value),
                                            feature_value
                                            )
            h = self.dense1(feature_value_masked)
            h2 = self.dense2(h)
            h3 = self.dense3(h2)
            return h3

        higher_input = inputs + random_v

        lower_input_maybe_neg = inputs - random_v
        is_pos_mask = tf.cast(tf.less(0.0, lower_input_maybe_neg), tf.float32)
        is_neg_or_zeo_mask = 1.0 - is_pos_mask
        lower_input = (lower_input_maybe_neg * is_pos_mask) + (inputs * is_neg_or_zeo_mask)

        h_higher = all_network(higher_input)
        h_lower = all_network(lower_input)
        h_out = all_network(inputs)
        h_diff1 = h_higher - h_out # This should be positive
        h_diff2 = h_out - h_lower # This should be positive
        increasing_loss1 = tf.reduce_mean(tf.math.maximum(-h_diff1, 0))
        increasing_loss2 = tf.reduce_mean(tf.math.maximum(-h_diff2, 0))
        self.add_loss(increasing_loss1)
        self.add_loss(increasing_loss2)
        return h_out


class ConvexLayer3(tf.keras.layers.Layer):
    def __init__(self, hidden_dim1=100, hidden_dim2=100, num_features=7):
        super(ConvexLayer3, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_dim1, activation='relu')
        self.dense2 = tf.keras.layers.Dense(hidden_dim2, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1)
        self.k = tf.Variable(tf.random_normal_initializer(0.01)([num_features]), trainable=True)

        self.dense_01 = []
        self.dense_02 = []
        for j in range(num_features):
            self.dense_01.append(tf.keras.layers.Dense(32, name=f"per_item1_{j}", activation='relu'))
            self.dense_02.append(tf.keras.layers.Dense(1, name=f"per_item2_{j}"))

    def call(self, inputs, **kwargs):
        is_used = tf.less(-0.5, inputs)
        shape = get_shape_list2(inputs)
        random_v = tf.abs(tf.random.normal(stddev=0.1, shape=shape))
        random_v = random_v * tf.cast(is_used, tf.float32)

        def all_network(x_input):
            is_used = tf.less(-0.5, x_input)
            x_is_used_f = tf.cast(is_used, tf.float32)
            h0 = []
            for j in range(num_features):
                x0_ex = tf.expand_dims(x_input[:, j], 1)
                x0_1 = self.dense_01[j](x0_ex)
                x0_1 = self.dense_02[j](x0_1)
                h0.append(x0_1)
            scaled_feature_value = tf.stack(h0, axis=1)
            x_is_used_f = tf.expand_dims(x_is_used_f, 2)
            all_features = tf.concat([scaled_feature_value, x_is_used_f], axis=2)
            h = self.dense1(all_features)
            h1 = tf.reduce_sum(h, 2)
            h2 = self.dense2(h1)
            h3 = self.dense3(h2)
            return h3

        higher_input = inputs + random_v
        lower_input_maybe_neg = inputs - random_v
        is_pos_mask = tf.cast(tf.less(0.0, lower_input_maybe_neg), tf.float32)
        is_neg_or_zeo_mask = 1.0 - is_pos_mask
        lower_input = (lower_input_maybe_neg * is_pos_mask) + (inputs * is_neg_or_zeo_mask)

        h_higher = all_network(higher_input)
        h_lower = all_network(lower_input)
        h_out = all_network(inputs)
        h_diff1 = h_higher - h_out # This should be positive
        h_diff2 = h_out - h_lower # This should be positive
        increasing_loss1 = tf.reduce_mean(tf.math.maximum(-h_diff1, 0))
        increasing_loss2 = tf.reduce_mean(tf.math.maximum(-h_diff2, 0))
        self.add_loss(increasing_loss1)
        self.add_loss(increasing_loss2)
        return h_out


class ConvexLayer4(tf.keras.layers.Layer):
    def __init__(self, num_features):
        super(ConvexLayer4, self).__init__()
        self.num_features = num_features
        self.k = tf.Variable(tf.random_normal_initializer(0.1)([num_features]), trainable=True)
        self.dense_01 = []
        self.dense_02 = []
        for j in range(num_features):
            self.dense_01.append(tf.keras.layers.Dense(32, name=f"per_item1_{j}", activation='sigmoid'))
            self.dense_02.append(tf.keras.layers.Dense(1, name=f"per_item2_{j}"))

    def call(self, inputs, **kwargs):
        is_used = tf.less(-0.5, inputs)
        shape = get_shape_list2(inputs)
        batch_size = shape[0]
        random_v = tf.abs(tf.random.normal(stddev=0.1, shape=shape))
        random_v = random_v * tf.cast(is_used, tf.float32)

        def log_filter_nan(x_in):
            x = tf.math.log(x_in)
            x_out = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
            return x_out

        def all_network(x_input):
            is_used = tf.less(-0.5, x_input)
            noise_amount = log_filter_nan(1 / 10 / x_input)
            x_is_used_f = tf.cast(is_used, tf.float32)
            h0 = []
            for j in range(self.num_features):
                noise_ex = tf.expand_dims(noise_amount[:, j], 1)
                noise_less_value = tf.reshape(self.k[j], [1, 1])
                nlv_ex = tf.tile(noise_less_value, [batch_size, 1])
                x0 = tf.concat([noise_ex, nlv_ex], axis=1)
                x0_1 = self.dense_01[j](x0)
                x0_1 = self.dense_02[j](x0_1)
                h0.append(x0_1)
            scaled_feature_value = tf.reshape(tf.stack(h0, axis=1), [-1, self.num_features])
            scaled_feature_value = scaled_feature_value * x_is_used_f
            return scaled_feature_value

        higher_input = inputs + random_v
        lower_input_maybe_neg = inputs - random_v
        is_pos_mask = tf.cast(tf.less(0.0, lower_input_maybe_neg), tf.float32)
        is_neg_or_zeo_mask = 1.0 - is_pos_mask
        lower_input = (lower_input_maybe_neg * is_pos_mask) + (inputs * is_neg_or_zeo_mask)
        h_higher = all_network(higher_input)
        h_lower = all_network(lower_input)
        h_out = all_network(inputs)
        h_diff1 = h_higher - h_out # This should be positive
        h_diff2 = h_out - h_lower # This should be positive

        def apply_loss_if_negative(h_diff):
            per_inst_dim_losses = tf.math.maximum(-h_diff, 0)
            return tf.reduce_mean(tf.reduce_sum(per_inst_dim_losses, axis=1)) * 10

        increasing_loss1 = apply_loss_if_negative(h_diff1)
        increasing_loss2 = apply_loss_if_negative(h_diff2)
        self.add_loss(increasing_loss1)
        self.add_loss(increasing_loss2)
        return h_out


def convert_tensor(items, num_features):
    def generator():
        for action, reward in items:
            feature = action
            label = reward
            yield feature, label

    return tf.data.Dataset.from_generator(generator,
                                   output_types=(tf.float32, tf.float32),
                                   output_shapes=([num_features,], [] ))


def get_model1(num_features, max_budget):
    print("get_model 1")
    x_input = tf.keras.Input([num_features, max_budget])
    x_is_used = tf.cast(tf.less(-0.5, x_input), tf.float32)
    x_new = tf.concat([x_input, x_is_used], axis=1)
    h = tf.keras.layers.Dense(100, activation='relu')(x_new)
    # x = tf.expand_dims(x_input, -1)
    # conv = tf.keras.layers.Conv1D(32, 1)
    # W = tf.Variable(np.array([1.0]), dtype=tf.float32)
    # y = conv(x)
    # y = tf.multiply(W, x)
    h = tf.keras.layers.Dense(20, activation='relu')(h)
    h = tf.keras.layers.Dense(20, activation='relu')(h)
    h = tf.keras.layers.Dense(1)(h)
    model = tf.keras.Model(inputs=[x_input], outputs=[h])
    return model


def get_model2(num_features, max_budget):
    x_input = tf.keras.Input([num_features])
    x_is_used = tf.cast(tf.less(-0.5, x_input), tf.float32)
    x_is_used = tf.expand_dims(x_is_used, -1)
    conv = tf.keras.layers.Conv1D(32, 1, input_shape=x_input.shape[1:])
    x_input_d3 = tf.expand_dims(x_input, -1)
    h0 = conv(x_input_d3)
    x_new = tf.concat([h0, x_is_used], axis=2)
    h = tf.keras.layers.Dense(100, activation='relu')(x_new)
    h1 = tf.reduce_sum(h, 2)
    h2 = tf.keras.layers.Dense(100, activation='relu')(h1)
    h3 = tf.keras.layers.Dense(1)(h2)
    model = tf.keras.Model(inputs=[x_input], outputs=[h3])
    return model


def get_model3(num_features, max_budget):
    x_input = tf.keras.Input([num_features])
    is_used = tf.less(-0.5, x_input)
    x_is_used_f = tf.cast(is_used, tf.float32)
    # x_input_norm = x_input + (1-tf.cast(x_is_used_f, tf.float32))

    h0 = []
    for j in range(num_features):
        x0_ex = tf.expand_dims(x_input[:, j], 1)
        x0_1 = tf.keras.layers.Dense(32, name=f"per_item1_{j}", activation='relu')(x0_ex)
        x0_1 = tf.keras.layers.Dense(1, name=f"per_item2_{j}")(x0_1)
        h0.append(x0_1)
    scaled_feature_value = tf.stack(h0, axis=1)
    x_is_used_f = tf.expand_dims(x_is_used_f, 2)
    all_features = tf.concat([scaled_feature_value, x_is_used_f], axis=2)
    h = tf.keras.layers.Dense(100, activation='relu')(all_features)
    h1 = tf.reduce_sum(h, 2)
    h2 = tf.keras.layers.Dense(100, activation='relu')(h1)
    h3 = tf.keras.layers.Dense(1)(h2)
    model = tf.keras.Model(inputs=[x_input], outputs=[h3])
    return model


def get_model4(num_features, max_budget):
    x_input = tf.keras.Input([num_features])
    is_used = tf.less(-0.5, x_input)
    x_is_used_f = tf.cast(is_used, tf.float32)
    x_input_norm = x_input + (1-tf.cast(x_is_used_f, tf.float32))
    x_is_used_f_ext = tf.expand_dims(x_is_used_f, -1)
    k = tf.Variable(tf.random_normal_initializer(0.01)([num_features]), trainable=True)

    x_input_norm_inv = 1 / x_input
    feature_value = -tf.math.log(tf.multiply(k, x_input_norm_inv))
    feature_value_masked = tf.where(tf.math.is_nan(feature_value),
                                    tf.zeros_like(feature_value),
                                    feature_value
                                    )

    h0 = []
    for j in range(num_features):
        x0_ex = tf.expand_dims(feature_value_masked[:, j], 1)
        x0_1 = tf.keras.layers.Dense(32, name=f"per_item1_{j}", activation='relu')(x0_ex)
        x0_1 = tf.keras.layers.Dense(1, name=f"per_item2_{j}")(x0_1)
        h0.append(x0_1)
    scaled_feature_value = tf.stack(h0, axis=1)
    all_features = tf.concat([scaled_feature_value,
                              x_is_used_f_ext,
                              tf.expand_dims(feature_value_masked, 2)], axis=2)
    # [batch, num_feature, 3]
    h = tf.keras.layers.Dense(100, activation='relu')(all_features)
    h1 = tf.reduce_sum(h, 2)
    h2 = tf.keras.layers.Dense(100, activation='relu')(h1)
    h3 = tf.keras.layers.Dense(1)(h2)
    model = tf.keras.Model(inputs=[x_input], outputs=[h3])
    return model


def get_model5(num_features, max_budget):
    print("model5")
    x_input = tf.keras.Input([num_features])
    is_used = tf.less(-0.5, x_input)
    k = tf.Variable(tf.random_normal_initializer(0.01)([num_features]), trainable=True)

    x_input_norm_inv = 1 / x_input
    feature_value = -tf.math.log(tf.multiply(k, x_input_norm_inv))
    feature_value_masked = tf.where(tf.math.is_nan(feature_value),
                                    tf.zeros_like(feature_value),
                                    feature_value
                                    )

    # [batch, num_feature, 3]
    h = tf.keras.layers.Dense(100, activation='relu')(feature_value_masked)
    h2 = tf.keras.layers.Dense(100, activation='relu')(h)
    h3 = tf.keras.layers.Dense(1)(h2)
    model = tf.keras.Model(inputs=[x_input], outputs=[h3])
    return model


def get_model6(num_features, max_budget):
    print("model6")
    x_input = tf.keras.Input([num_features]) # This is less than 1
    x_input_normalized = 1 / max_budget * x_input
    is_used = tf.less(-0.5, x_input_normalized)
    k = tf.Variable(tf.random_normal_initializer(0.01)([num_features]), trainable=True)

    x_input_norm_inv = 1 / x_input_normalized # This is larger than 1
    # When k = input, feature_value = 0
    feature_value = -tf.math.log(tf.multiply(k, x_input_norm_inv))
    feature_value_masked = tf.where(tf.math.is_nan(feature_value),
                                    tf.zeros_like(feature_value),
                                    feature_value
                                    )

    # [batch, num_feature, 3]
    h3 = ConvexLayer()(feature_value_masked)
    model = tf.keras.Model(inputs=[x_input], outputs=[h3])
    return model


def get_model7(num_features, max_budget):
    print("model 7")
    x_input = tf.keras.Input([num_features]) # This is less than 1
    network = ConvexLayer2()
    h3 = network(x_input)
    model = tf.keras.Model(inputs=[x_input], outputs=[h3])
    return model


def get_model8(num_features, max_budget):
    print("model 8")
    x_input = tf.keras.Input([num_features])
    x_input_norm = x_input * (1 / max_budget)
    layer = ConvexLayer4(num_features)
    feature_value = layer(x_input_norm)
    # [batch, num_feature, 3]
    h = tf.keras.layers.Dense(100, activation='relu')(feature_value)
    h2 = tf.keras.layers.Dense(100, activation='relu')(h)
    h3 = tf.keras.layers.Dense(1)(h2)
    model = tf.keras.Model(inputs=[x_input], outputs=[h3])
    return model


def get_model9(num_features, max_budget):
    print("model 9")
    x_input = tf.keras.Input([num_features])
    layer = ConvexLayer(num_features)

    h = layer(x_input)
    h2 = tf.keras.layers.Dense(100, activation='relu')(h)
    h2 = tf.keras.layers.Dense(100, activation='relu')(h2)
    h3 = tf.keras.layers.Dense(1)(h2)
    model = tf.keras.Model(inputs=[x_input], outputs=[h3])
    return model


def get_model10(num_features, max_budget):
    print("model 10")
    x_input = tf.keras.Input([num_features])
    h = tf.keras.layers.Dense(100, activation='relu')(x_input)
    h2 = tf.keras.layers.Dense(100, activation='relu')(h)
    h3 = tf.keras.layers.Dense(1)(h2)
    model = tf.keras.Model(inputs=[x_input], outputs=[h3])
    return model


def get_model11(num_features, max_budget):
    print("model 11")
    x_input = tf.keras.Input([num_features])
    r_out_list = []
    r_layers = []
    is_negative = tf.less(x_input, 0)
    is_positive_mask = tf.cast(tf.logical_not(is_negative), tf.float32)
    for i in range(num_features):
        r_layer = RegressionLayer("regression_{}".format(i))
        r_layers.append(r_layer)
        r_out = r_layer(x_input[:, i:i+1])
        r_out_list.append(r_out)

    r_out_h = tf.stack(r_out_list, axis=1)
    r_out_h = is_positive_mask * r_out_h
    layer = ConvexLayer(num_features)
    h = layer(r_out_h)
    h2 = tf.keras.layers.Dense(100, activation='relu')(h)
    h2 = tf.keras.layers.Dense(100, activation='relu')(h2)
    h3 = tf.keras.layers.Dense(1)(h2)
    model = tf.keras.Model(inputs=[x_input], outputs=[h3])
    return model
