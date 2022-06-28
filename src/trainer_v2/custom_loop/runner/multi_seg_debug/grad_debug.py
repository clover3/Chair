import code

import tensorflow as tf

from trainer_v2.custom_loop.neural_network_def.segmented_enc import FuzzyLogicLayerM


def get_dummy_inputs():
    batch_size = 4
    seq_len = 10
    input_mask = tf.constant([[1] * 6 + [0] * 4] * batch_size)
    random_logits = tf.random.uniform([batch_size, seq_len, 3])
    local_decisions = tf.nn.softmax(random_logits, axis=-1)
    y = tf.ones([batch_size], tf.int32)
    return input_mask, local_decisions, y


def get_dummy_inputs():
    batch_size = 4
    seq_len = 10
    input_mask = tf.constant([[0] * 5 + [1, 1, 1, 1, 1]] * batch_size)
    print(input_mask)
    random_logits = tf.random.uniform([batch_size, seq_len, 3])
    local_decisions = tf.nn.softmax(random_logits, axis=-1)
    y = tf.ones([batch_size], tf.int32)
    return input_mask, local_decisions, y


def main():
    with tf.GradientTape(watch_accessed_variables=True, persistent=True) as tape:
        input_mask, local_decisions, y = get_dummy_inputs()
        tape.watch(local_decisions)
        input_mask_ex = tf.cast(tf.expand_dims(input_mask, 2), tf.float32)
        local_decisions_masked = tf.multiply(local_decisions, input_mask_ex)

        input_mask_f = tf.cast(input_mask, tf.float32)

        local_entail_p = local_decisions_masked[:, :, 0]
        local_neutral_p = local_decisions_masked[:, :, 1]
        local_contradiction_p = local_decisions_masked[:, :, 2]
        combined_contradiction_s = tf.reduce_max(local_contradiction_p, axis=-1)

        cnp1 = tf.reduce_max(local_neutral_p, axis=-1)  # [batch_size]
        cnp2 = 1 - combined_contradiction_s
        combined_neutral_s = tf.multiply(cnp1, cnp2)
        eps = 1e-6

        def mean(t, axis):
            t2 = tf.reduce_sum(t, axis)
            n_valid = tf.cast(tf.reduce_sum(input_mask_f, axis=1), tf.float32) + eps
            return tf.divide(t2, n_valid)

        log_local_entail_p = tf.math.log(local_entail_p + eps) * input_mask_f
        mean_log_local_entail_p = mean(log_local_entail_p, -1)
        combined_entail_s = tf.math.exp(mean_log_local_entail_p)
        score_stacked = tf.stack([combined_entail_s, combined_neutral_s, combined_contradiction_s], axis=1)
        sum_s = tf.reduce_sum(score_stacked, axis=1, keepdims=True)
        sentence_logits = tf.divide(score_stacked, sum_s)
        sentence_prob = tf.nn.softmax(sentence_logits, axis=1)

        ce = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        loss = ce(y, sentence_prob)

    vars = [local_decisions, local_decisions_masked, local_entail_p, log_local_entail_p,
            mean_log_local_entail_p,
            combined_entail_s, sentence_logits]

    gradient = tape.gradient(loss, vars)
    for idx, (v, g) in enumerate(zip(vars, gradient)):
        print("Idx", idx)
        print("Var", v)
        print("Grad", g)



def main2():
    with tf.GradientTape(watch_accessed_variables=True, persistent=True) as tape:
        input_mask, local_decisions, y = get_dummy_inputs()
        tape.watch(local_decisions)
        tape.watch(input_mask)
        sentence_prob = FuzzyLogicLayerM()(local_decisions, input_mask)
        ce = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        loss = ce(y, sentence_prob)

    vars = [local_decisions, input_mask]

    gradient = tape.gradient(loss, vars)
    for idx, (v, g) in enumerate(zip(vars, gradient)):
        print("Idx", idx)
        print("Var", v)
        print("Grad", g)
    code.interact(local=locals())



if __name__ == "__main__":
    main()