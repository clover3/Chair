import tensorflow as tf


def brutal_loss_compare(features):
    loss1 = features["loss1"]
    loss2 = features["loss2"]

    prob1 = tf.exp(-loss1)
    prob2 = tf.exp(-loss2)

    output = -(prob1 - prob2)
    return output


def blc_beta(features):
    def t_init(t):
        t = tf.constant(t)
        t = tf.expand_dims(t, 0)
        t = tf.expand_dims(t, 0)
        return t
    
    mean_list = t_init([0.00592526, 0.046476, 0.06966591, 0.0852695, 0.0819463, 0.11604176
                            , 0.13313128, 0.14524656, 0.17160586, 0.18507012, 0.19524361, 0.23223796
                            , 0.23867801, 0.2618752, 0.28670366, 0.31369072, 0.3431509, 0.39701927
                            , 0.45573084, 0.72065012, 0.99999865])
    std_list = t_init([3.25959660e-02, 8.10599402e-02, 1.18821315e-01, 1.27397642e-01
                    , 1.20389193e-01, 1.51040196e-01, 1.76706269e-01, 1.86607167e-01
                    , 1.98343635e-01, 2.06294343e-01, 2.20862463e-01, 2.48731196e-01
                    , 2.37961769e-01, 2.49380141e-01, 2.60327846e-01, 2.84994602e-01
                    , 2.89982408e-01, 3.05399150e-01, 3.16005856e-01, 3.12512785e-01
                    , 2.93719731e-05])
    st_list = t_init([0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65
                            , 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1., ])
    ed_list = t_init([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7
                            , 0.75, 0.8, 0.85, 0.9, 0.95, 1., 1.05])

    def get_scores_lin(prob1_list, prob2_list):
        v2 = tf.reduce_min(tf.stack([prob1_list, prob2_list], axis=-1), axis=-1)
        v2 = tf.expand_dims(v2, -1) #[batch, seq_len, 1]
        all_scores = (v2 - mean_list) / std_list
        prob1_list = tf.expand_dims(prob1_list, -1)
        f1 = tf.less_equal(st_list, prob1_list)
        f2 = tf.less(prob1_list, ed_list)
        f = tf.logical_and(f1, f2)
        all_scores = all_scores * tf.cast(f, tf.float32)
        scores = tf.reduce_sum(all_scores, axis=-1)
        return scores

    loss1 = features["loss1"]
    loss2 = features["loss2"]

    prob1 = tf.exp(-loss1)
    prob2 = tf.exp(-loss2)
    return get_scores_lin(prob1, prob2)
