from __future__ import print_function
import tensorflow as tf

from models.transformer.modules import *
import os, codecs


def transformer_encode(x, hp, voca_size, is_training):
    with tf.variable_scope("encoder"):
        ## Embedding
        enc = embedding(x,
                         vocab_size=voca_size,
                         num_units=hp.hidden_units,
                         scale=True,
                         scope="enc_embed")

        ## Positional Encoding
        if hp.sinusoid:
            enc += positional_encoding(x,
                                        num_units=hp.hidden_units,
                                        zero_pad=False,
                                        scale=False,
                                        scope="enc_pe")
        else:
            enc += embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(x)[1]), 0), [tf.shape(x)[0], 1]),
                vocab_size=hp.seq_max,
                num_units=hp.hidden_units,
                zero_pad=False,
                scale=False,
                scope="enc_pe")

        ## Dropout
        enc = tf.layers.dropout(enc,
                                 rate=hp.dropout_rate,
                                 training=tf.convert_to_tensor(is_training))

        ## Blocks
        for i in range(hp.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i)):
                ### Multihead Attention
                enc = multihead_attention(queries=enc,
                                               keys=enc,
                                               num_units=hp.hidden_units,
                                               num_heads=hp.num_heads,
                                               dropout_rate=hp.dropout_rate,
                                               is_training=is_training,
                                               causality=False)

                ### Feed Forward
                enc = feedforward(enc, num_units=[4 * hp.hidden_units, hp.hidden_units])

    return enc

