import os

import numpy as np
import tensorflow as tf

import cpath
from bert_api.bert_masking_common import BERTMaskIF
from models.bert_util.bert_utils import batch2feed_dict_4_or_5_inputs, get_last_id
from models.transformer import hyperparams
from models.transformer.bert_common_v2 import create_attention_mask_from_input_mask
from tf_v2_support import placeholder, disable_eager_execution, tf1
from tlm.model import base as bert
from tlm.model.bert_mask import apply_drop, BertModelMasked


def init_session():
    config = tf1.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False)
    config.gpu_options.allow_growth = True

    return tf1.Session(config=config)


def get_batches_ex(data, batch_size, n_inputs):
    # data is fully numpy array here
    step_size = int((len(data) + batch_size - 1) / batch_size)
    new_data = []
    for step in range(step_size):
        b_unit = [list() for i in range(n_inputs)]

        for i in range(batch_size):
            idx = step * batch_size + i
            if idx >= len(data):
                break
            for input_i in range(n_inputs):
                b_unit[input_i].append(data[idx][input_i])
        if len(b_unit[0]) > 0:
            batch = [np.stack(b_unit[input_i]) for input_i in range(n_inputs)]
            new_data.append(batch)

    return new_data


class transformer_attn_mask:
    def __init__(self, hp, num_classes, voca_size, is_training=True):
        config = bert.BertConfig(vocab_size=voca_size,
                                 hidden_size=hp.hidden_units,
                                 num_hidden_layers=hp.num_blocks,
                                 num_attention_heads=hp.num_heads,
                                 intermediate_size=hp.intermediate_size,
                                 type_vocab_size=hp.type_vocab_size,
                                 )

        seq_length = hp.seq_max
        use_tpu = False

        input_ids = placeholder(tf.int64, [None, seq_length])
        input_mask = placeholder(tf.int64, [None, seq_length])
        segment_ids = placeholder(tf.int64, [None, seq_length])
        attention_drop_mask = placeholder(tf.int64, [None, seq_length, seq_length])
        label_ids = placeholder(tf.int64, [None])
        self.x_list = [input_ids, input_mask, segment_ids, attention_drop_mask]
        self.y = label_ids

        attention_mask = create_attention_mask_from_input_mask(
            input_ids, input_mask)
        attention_mask = apply_drop(attention_drop_mask, attention_mask)
        self.attention_mask = attention_mask

        use_one_hot_embeddings = use_tpu
        self.model = BertModelMasked(
            config=config,
            is_training=is_training,
            attention_mask=attention_mask,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        pooled_output = self.model.get_pooled_output()
        logits = tf.keras.layers.Dense(num_classes, name="cls_dense")(pooled_output)
        loss_arr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=label_ids)
        loss = tf.reduce_mean(input_tensor=loss_arr)

        self.loss = loss
        self.logits = logits

    def batch2feed_dict(self, batch):
        return batch2feed_dict_4_or_5_inputs(self, batch)


class PredictorAttentionMask(BERTMaskIF):
    def __init__(self, num_classes, seq_len=None):
        disable_eager_execution()
        self.voca_size = 30522
        self.hp = hyperparams.HPFAD()
        if seq_len is not None:
            self.hp.seq_max = seq_len
        self.model_dir = cpath.common_model_dir_root
        self.task = transformer_attn_mask(self.hp, num_classes, self.voca_size, False)
        self.sess = init_session()
        self.sess.run(tf1.global_variables_initializer())
        self.batch_size = 32
        self.model_loaded = False

    def predict(self, payload):
        if not self.model_loaded:
            print("WARNING model has not been loaded")
        def forward_run(inputs):
            batches = get_batches_ex(inputs, self.batch_size, 4)
            logit_list = []
            for batch in batches:
                logits,  = self.sess.run([self.task.logits, ],
                                         feed_dict=self.task.batch2feed_dict(batch))
                logit_list.append(logits)
            return np.concatenate(logit_list)

        payload = [self.unpack_dict(e) for e in payload]
        scores = forward_run(payload)
        return scores.tolist()

    def unpack_dict(self, payload_item):
        max_seq_length = self.hp.seq_max
        x0, x1, x2, x3 = payload_item
        x3_np = np.zeros([max_seq_length, max_seq_length])
        for k, v in x3.items():
            idx1, idx2 = k
            x3_np[idx1, idx2] = v
        return np.array(x0), np.array(x1), np.array(x2), x3_np

    def load_model(self, save_dir):
        id = get_last_id(save_dir)
        path = os.path.join(save_dir, "{}".format(id))
        print("load_model")
        self.loader = tf1.train.Saver(max_to_keep=1)
        self.loader.restore(self.sess, path)
        self.model_loaded = True