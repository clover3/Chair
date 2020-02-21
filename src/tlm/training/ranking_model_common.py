import tensorflow as tf


def combine_paired_input_features(features):
    input_ids1 = features["input_ids1"]
    input_mask1 = features["input_mask1"]
    segment_ids1 = features["segment_ids1"]

    # Negative Example
    input_ids2 = features["input_ids2"]
    input_mask2 = features["input_mask2"]
    segment_ids2 = features["segment_ids2"]

    input_ids = tf.concat([input_ids1, input_ids2], axis=0)
    input_mask = tf.concat([input_mask1, input_mask2], axis=0)
    segment_ids = tf.concat([segment_ids1, segment_ids2], axis=0)
    return input_ids, input_mask, segment_ids


def pairwise_model(pooled_output):
    logits = tf.keras.layers.Dense(1, name="cls_dense")(pooled_output)
    pair_logits = tf.reshape(logits, [2, -1])
    y_pred = pair_logits[0, :] - pair_logits[1, :]
    losses = tf.maximum(1.0 - y_pred, 0)
    loss = tf.reduce_mean(losses)
    return loss, losses, y_pred


def pairwise_cross_entropy(pooled_output):
    logits = tf.keras.layers.Dense(1, name="cls_dense")(pooled_output)
    pair_logits = tf.reshape(logits, [2, -1])
    prob = tf.nn.softmax(pair_logits, axis=0)
    losses = 1 - prob[0, :]
    loss = tf.reduce_mean(losses)
    return loss, losses, prob[0, :]


def cross_entropy(pooled_output):
    logits = tf.keras.layers.Dense(2, name="cls_dense")(pooled_output)
    real_batch_size = tf.cast(logits.shape[0] / 2, tf.int32)

    labels = tf.concat([tf.ones([real_batch_size], tf.int32),
                         tf.zeros([real_batch_size], tf.int32)], axis=0)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels)
    loss = tf.reduce_mean(losses)
    return loss, losses, tf.reshape(logits, [2, -1, 2])[0, :, 1]


def multilabel_hinge_model(pooled_output, features):
    num_label = 3
    logits = tf.keras.layers.Dense(num_label, name="cls_dense")(pooled_output)
    pair_logits = tf.reshape(logits, [2, -1, num_label])
    label_ids1 = features['label_ids1']
    label_ids2 = features['label_ids2']

    all_losses = []
    for i in range(num_label):
        # In our first implementation, we make a,b to be different label, otherwise it is waste of computation
        a = tf.equal(label_ids1, i)  # Typically this is positive entry, 1/3 of entries will be true)
        b = tf.not_equal(label_ids2, i)  # If a is true, b is supposed be true, but, just in case

        valid = tf.cast(tf.logical_and(a, b), tf.float32)
        y_pred = pair_logits[0, :, i] - pair_logits[1, :, i]
        losses = tf.maximum(1.0 - y_pred, 0) * valid
        all_losses.append(losses)

    losses = all_losses[0]
    for i in range(1,num_label):
        losses += all_losses[i]

    loss = tf.reduce_mean(losses)
    all_pred = pair_logits[0] - pair_logits[1]
    return loss, losses, all_pred


def get_prediction_structure(modeling_opt, pooled_output):
    if modeling_opt == "hinge" or modeling_opt == "pair_ce":
        logits = tf.keras.layers.Dense(1, name="cls_dense")(pooled_output)
    elif modeling_opt == "ce":
        raw_logits = tf.keras.layers.Dense(2, name="cls_dense")(pooled_output)
        probs = tf.nn.softmax(raw_logits, axis=1)
        logits = probs[:, 1]
    elif modeling_opt == "multi_label_hinge":
        logits = tf.keras.layers.Dense(3, name="cls_dense")(pooled_output)
    else:
        assert False
    return logits


def apply_loss_modeling(modeling_opt, pooled_output, features):
    if modeling_opt == "hinge":
        loss, losses, y_pred = pairwise_model(pooled_output)
    elif modeling_opt == "pair_ce":
        loss, losses, y_pred = pairwise_cross_entropy(pooled_output)
    elif modeling_opt == "ce":
        loss, losses, y_pred = cross_entropy(pooled_output)
    elif modeling_opt == "all_pooling":
        loss, losses, y_pred = cross_entropy(pooled_output)
    elif modeling_opt == "multi_label_hinge":
        loss, losses, y_pred = multilabel_hinge_model(pooled_output, features)
    else:
        assert False
    return loss, losses, y_pred