from typing import List

import tensorflow as tf

from trainer_v2.per_project.transparency.mmp.data_gen.tt_train_gen import get_convert_to_bow_qtw


def pairwise_hinge(score_pos, score_neg):
    loss_cont = tf.maximum(1 - (score_pos - score_neg), 0)
    return loss_cont


def get_tt_scorer(tti: tf.keras.models.Model):
    """

    :param tti: Model that scores q, d
    :return:
    """
    SpecI = tf.TensorSpec([None], dtype=tf.int32)
    SpecF = tf.TensorSpec([None], dtype=tf.float32)
    convert_to_bow = get_convert_to_bow_qtw()
    sig = (SpecI, SpecI, SpecF, SpecI, SpecI, SpecF),

    def score_fn(qd_list: List):
        def generator():
            for query, document in qd_list:
                x = []
                for text in [query, document]:
                    tfs, input_ids, qtw = convert_to_bow(text)
                    for l in [input_ids, tfs, qtw]:
                        x.append(tf.constant(l))
                yield tuple(x),

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=sig)
        dataset = dataset.batch(16)
        output = tti.predict(dataset)
        return output

    return score_fn


