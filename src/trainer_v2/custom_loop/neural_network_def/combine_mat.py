import tensorflow as tf
from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.chair_logging import c_log


def cpt_combine(local_decisions):
    mat = tf.constant([[0, 1, 2],
                       [1, 1, 2],
                       [2, 2, 2]])

    local_decision_a = local_decisions[:, 0]
    local_decision_b = local_decisions[:, 1]  # [B, 3]

    mat_one_hot = tf.one_hot(mat, 3)  # [3, 3, 3]   axis 2 is one hot
    mat_f_right = tf.reshape(mat_one_hot, [3, 9])
    a_mask = tf.matmul(local_decision_a, mat_f_right)  # [B, 9]
    m = tf.reshape(a_mask, [-1, 3, 3])  # [B, 3, 3] axis 2 is one hot
    # m = tf.transpose(a_mask2, [0, 2, 1])  # [B, 3, 3] axis 1 is for classes
    print('m', m.shape)
    print('local_decision_b', local_decision_b.shape)
    result = tf.reduce_sum(tf.multiply(m, tf.expand_dims(local_decision_b, axis=2)), axis=1)
    # result = tf.matmul(m, local_decision_b, transpose_b=True)  # [B, 3]
    print('result', result.shape)
    result = tf.reshape(result, [-1, 3]) # [B, 3, 1]
    print('result', result.shape)
    return result


# CPT: Conditional Probability Table
def cpt_combine2(local_decisions):
    cpt_discrete = tf.constant([[0, 1, 2],
                       [1, 1, 2],
                       [2, 2, 2]])

    local_decision_a = local_decisions[:, 0]
    local_decision_b = local_decisions[:, 1]  # [B, 3]

    cpt = tf.one_hot(cpt_discrete, 3)  # [3, 3, 3]   axis 2 is one hot

    left = tf.expand_dims(tf.expand_dims(local_decision_a, 2), 3)  # [B, 3, 1, 1]
    right = tf.expand_dims(cpt, axis=0)  # [1, 3, 3, 3]
    t = tf.multiply(left, right)
    res1 = tf.reduce_sum(t, axis=1)  # [B, 3, 3]

    left = tf.expand_dims(local_decision_b, axis=2)  #[B, 3, 1]
    right = res1
    t = tf.multiply(left, right)
    result = tf.reduce_sum(t, axis=1)  # [B, 3]
    return result


def cpt_combine3(local_decisions):
    mat = tf.constant([[0, 1, 2],
                       [1, 1, 2],
                       [2, 2, 2]])
    local_decision_a = local_decisions[:, 0]
    local_decision_b = local_decisions[:, 1]  # [B, 3]
    return cpt_combine_var(local_decision_a, local_decision_b, mat)


def cpt_combine_var(local_decision_a, local_decision_b, mat):
    left = tf.expand_dims(local_decision_a, 2)
    right = tf.expand_dims(local_decision_b, 1)
    co_prob = tf.multiply(left, right)  # [B, 3, 3]
    mat_one_hot = tf.one_hot(mat, 3)  # [3, 3, 3]   axis 2 is one hot
    t = tf.tensordot(co_prob, mat_one_hot, axes=[[1, 2], [0, 1]]) # [B, 3, 3]
    return t


def cpt_combine4(local_decisions):
    mat = tf.constant([[0, 1, 1, 2],
                       [1, 1, 1, 2],
                       [1, 1, 1, 1],
                       [2, 2, 1, 2],
                       ])
    local_decision_a = local_decisions[:, 0]
    local_decision_b = local_decisions[:, 1]  # [B, 4]
    return cpt_combine_var(local_decision_a, local_decision_b, mat)


class MatrixCombine(tf.keras.layers.Layer):
    def call(self, inputs, *args, **kwargs):
        return cpt_combine2(inputs)


class MatrixCombine4(tf.keras.layers.Layer):
    def call(self, inputs, *args, **kwargs):
        return cpt_combine4(inputs)



def test_discrete():
    mat = tf.constant([[0, 1, 2],
                       [1, 1, 2],
                       [2, 2, 2]])

    for label_a in range(3):
        for label_b in range(3):
            a = tf.one_hot([label_a], 3)
            b = tf.one_hot([label_b], 3)
            gold = mat[label_a, label_b]
            local_decisions = tf.stack([a, b], axis=1)
            result = cpt_combine2(local_decisions)
            result3 = cpt_combine3(local_decisions)

            pred = tf.argmax(result[0])
            pred3 = tf.argmax(result3[0])
            gold_n = gold.numpy()
            pred_n = pred.numpy()
            pred3_n = pred3.numpy()
            print(label_a, label_b, gold_n, pred_n, gold_n == pred_n, pred_n == pred3_n)


def test_continuous():
    a = tf.constant([[0., 0.3, 0.7], [0., 0.3, 0.7]])
    b = tf.constant([[0.7, 0.3, 0.], [0., 0.3, 0.7]])
    local_decisions = tf.stack([a, b], axis=1)
    result = cpt_combine2(local_decisions)
    result3 = cpt_combine3(local_decisions)
    print(result)
    print(result - result3)


def test_discrete4():
    mat = tf.constant([[0, 1, 1, 2],
                       [1, 1, 1, 2],
                       [1, 1, 1, 1],
                       [2, 2, 1, 2],
                       ])

    for label_a in range(4):
        for label_b in range(4):
            a = tf.one_hot([label_a], 4)
            b = tf.one_hot([label_b], 4)
            gold = mat[label_a, label_b]
            local_decisions = tf.stack([a, b], axis=1)
            result = cpt_combine4(local_decisions)

            pred = tf.argmax(result[0])
            gold_n = gold.numpy()
            pred_n = pred.numpy()
            print(label_a, label_b, gold_n, pred_n, gold_n == pred_n)


if __name__ == "__main__":
    test_discrete4()
