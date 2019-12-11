import os
from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow import keras

from path import data_path
from tf_util.tf_logging import tf_logging
from tlm.dictionary.dict_reader_transformer import DictReaderModel
from tlm.dictionary.sense_selecting_dictionary_reader import SSDR
from tlm.model.base import BertConfig
from tlm.training.train_flags import FLAGS
from trainer import tf_module


class DictReaderInterface(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_cls_loss(self):
        pass

    @abstractmethod
    def get_cls_loss_arr(self):
        pass

    def get_lookup_logits(self):
        return self.lookup_logits

    @abstractmethod
    def get_lookup_loss(self):
        pass

    @abstractmethod
    def batch2feed_dict(self, batch):
        pass

    @abstractmethod
    def get_acc(self):
        pass

    @abstractmethod
    def get_p_at_1(self):
        pass


class DictReaderWrapper(DictReaderInterface):
    def __init__(self, num_classes, seq_length, is_training):
        super(DictReaderWrapper, self).__init__()
        placeholder = tf.compat.v1.placeholder
        bert_config = BertConfig.from_json_file(os.path.join(data_path, "bert_config.json"))
        def_max_length = FLAGS.max_def_length
        loc_max_length = FLAGS.max_loc_length
        tf_logging.debug("DictReaderWrapper init()")
        tf_logging.debug("seq_length %d" % seq_length)
        tf_logging.debug("def_max_length %d" % def_max_length)
        tf_logging.debug("loc_max_length %d" % loc_max_length)

        self.input_ids = placeholder(tf.int64, [None, seq_length])
        self.input_mask_ = placeholder(tf.int64, [None, seq_length])
        self.segment_ids = placeholder(tf.int64, [None, seq_length])

        self.d_input_ids = placeholder(tf.int64, [None, def_max_length])
        self.d_input_mask = placeholder(tf.int64, [None, def_max_length])
        self.d_location_ids = placeholder(tf.int64, [None, loc_max_length])

        self.y_cls = placeholder(tf.int64, [None])
        self.y_lookup = placeholder(tf.int64, [None, seq_length])

        self.network = DictReaderModel(
                config=bert_config,
                d_config=bert_config,
                is_training=is_training,
                input_ids=self.input_ids,
                input_mask=self.input_mask_,
                d_input_ids=self.d_input_ids,
                d_input_mask=self.d_input_mask,
                d_location_ids=self.d_location_ids,
                use_target_pos_emb=True,
                token_type_ids=self.segment_ids,
                use_one_hot_embeddings=False,
            )

        self.cls_logits = keras.layers.Dense(num_classes)(self.network.pooled_output)
        self.cls_loss_arr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.cls_logits,
            labels=self.y_cls)
        self.cls_loss = tf.reduce_mean(self.cls_loss_arr)

        self.lookup_logits = keras.layers.Dense(2)(self.network.sequence_output)
        self.lookup_loss_arr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.lookup_logits,
            labels=self.y_lookup)
        self.lookup_loss_per_example = tf.reduce_mean(self.lookup_loss_arr, axis=-1)
        self.lookup_loss = tf.reduce_mean(self.lookup_loss_per_example)
        self.acc = tf_module.accuracy(self.cls_logits, self.y_cls)

    def get_cls_loss(self):
        return self.cls_loss

    def get_cls_loss_arr(self):
        return self.cls_loss_arr

    def get_lookup_loss(self):
        return self.lookup_loss

    def batch2feed_dict(self, batch):
        x0, x1, x2, x3, x4, x5, y= batch
        feed_dict = {
            self.input_ids: x0,
            self.input_mask_: x1,
            self.segment_ids: x2,
            self.d_input_ids: x3,
            self.d_input_mask: x4,
            self.d_location_ids: x5,
            self.y_cls: y,
        }
        return feed_dict

    def get_acc(self):
        return self.acc


# [Batch, loc_max_length]
def get_y_lookup_from_location_ids(location_ids, seq_length):
    del_mask = tf.cast(tf.not_equal(location_ids, 0), tf.int64) # [batch, loc_max_length] 0 is used as padding, it is invalid index
    z = tf.one_hot(location_ids, depth=seq_length, dtype=tf.int64) # [batch, loc_max_length, seq_length]  0,1
    z = z * tf.expand_dims(del_mask, 2)
    z = tf.reduce_sum(z, axis=1) # [batch, seq_length]
    z = tf.cast(tf.not_equal(z, 0), tf.int64) # There should be nothing larger than 1, but just make sure
    return z

# Word Sense Selecting Dictionary Reader
class WSSDRWrapper(DictReaderInterface):
    def __init__(self, num_classes, ssdr_config, seq_length, is_training):
        super(WSSDRWrapper, self).__init__()
        placeholder = tf.compat.v1.placeholder
        bert_config = BertConfig.from_json_file(os.path.join(data_path, "bert_config.json"))
        def_max_length = FLAGS.max_def_length
        loc_max_length = FLAGS.max_loc_length
        tf_logging.debug("WSSDRWrapper init()")
        tf_logging.debug("seq_length %d" % seq_length)
        tf_logging.debug("def_max_length %d" % def_max_length)
        tf_logging.debug("loc_max_length %d" % loc_max_length)

        self.input_ids = placeholder(tf.int64, [None, seq_length], name="input_ids")
        self.input_mask_ = placeholder(tf.int64, [None, seq_length], name="input_mask")
        self.segment_ids = placeholder(tf.int64, [None, seq_length], name="segment_ids")
        self.d_location_ids = placeholder(tf.int64, [None, loc_max_length], name="d_location_ids")

        self.d_input_ids = placeholder(tf.int64, [None, def_max_length], name="d_input_ids")
        self.d_input_mask = placeholder(tf.int64, [None, def_max_length], name="d_input_mask")
        self.d_segment_ids = placeholder(tf.int64, [None, def_max_length], name="d_segment_ids")
        self.ab_mapping = placeholder(tf.int64, [None, 1], name="ab_mapping")
        if ssdr_config.use_ab_mapping_mask:
            self.ab_mapping_mask = placeholder(tf.int64, [None, FLAGS.def_per_batch], name="ab_mapping_mask")
        else:
            self.ab_mapping_mask = None

        # [batch,seq_len], 1 if the indices in d_locations_id
        y_lookup = get_y_lookup_from_location_ids(self.d_location_ids, seq_length)

        self.y_cls = placeholder(tf.int64, [None])

        self.network = SSDR(
                config=bert_config,
                ssdr_config=ssdr_config,
                is_training=is_training,
                input_ids=self.input_ids,
                input_mask=self.input_mask_,
                token_type_ids=self.segment_ids,
                d_input_ids=self.d_input_ids,
                d_input_mask=self.d_input_mask,
                d_segment_ids=self.d_segment_ids,
                d_location_ids=self.d_location_ids,
                ab_mapping=self.ab_mapping,
                ab_mapping_mask=self.ab_mapping_mask,
                use_one_hot_embeddings=False,
            )
        self.cls_logits = keras.layers.Dense(num_classes)(self.network.get_pooled_output())
        self.cls_loss_arr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.cls_logits,
            labels=self.y_cls)
        self.cls_loss = tf.reduce_mean(self.cls_loss_arr)

        self.lookup_logits = keras.layers.Dense(2)(self.network.get_sequence_output())
        self.lookup_p_at_1 = tf_module.p_at_1(self.lookup_logits[:, 1], y_lookup)
        self.lookup_loss_arr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.lookup_logits,
            labels=y_lookup)
        self.y_lookup = y_lookup
        self.lookup_loss_per_example = tf.reduce_sum(self.lookup_loss_arr, axis=-1)
        self.lookup_loss = tf.reduce_mean(self.lookup_loss_per_example)
        self.acc = tf_module.accuracy(self.cls_logits, self.y_cls)

    def get_cls_loss(self):
        return self.cls_loss

    def get_cls_loss_arr(self):
        return self.cls_loss_arr

    def get_lookup_loss(self):
        return self.lookup_loss

    def batch2feed_dict(self, batch):
        if self.ab_mapping_mask is None:
            return self._batch2feed_dict(batch)
        else:
            return self._batch2feed_dict_w_abmapping(batch)

    def _batch2feed_dict(self, batch):
        x0, x1, x2, x3, y, x4, x5, x6, ab_map = batch
        feed_dict = {
            self.input_ids: x0,
            self.input_mask_: x1,
            self.segment_ids: x2,
            self.d_location_ids: x3,
            self.d_input_ids: x4,
            self.d_input_mask: x5,
            self.d_segment_ids: x6,
            self.ab_mapping: ab_map,
            self.y_cls: y,
        }
        return feed_dict

    def _batch2feed_dict_w_abmapping(self, batch):
        x0, x1, x2, x3, y, x4, x5, x6, ab_map, ab_mapping_mask = batch

        feed_dict = {
            self.input_ids: x0,
            self.input_mask_: x1,
            self.segment_ids: x2,
            self.d_location_ids: x3,
            self.d_input_ids: x4,
            self.d_input_mask: x5,
            self.d_segment_ids: x6,
            self.ab_mapping: ab_map,
            self.ab_mapping_mask: ab_mapping_mask,
            self.y_cls: y,
        }
        return feed_dict

    def get_acc(self):
        return self.acc

    def get_p_at_1(self):
        return self.lookup_p_at_1
