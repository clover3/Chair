import os

import numpy as np
import tensorflow as tf

import models.bert_util.bert_utils
from cpath import output_path
from bert_api.bert_mask_predictor import get_batches_ex, PredictorAttentionMask


class PredictorAttentionMaskGradient(PredictorAttentionMask):
    def __init__(self, num_classes, seq_len=None):
        super(PredictorAttentionMaskGradient, self).__init__(num_classes, seq_len)
        logits = [self.task.logits[:, class_i] for class_i in range(num_classes)]
        attention_gradient = tf.gradients(logits, self.task.attention_mask)
        # List [batch_size, seq_length, seq_length]
        self.attention_gradient = tf.stack(attention_gradient, -1)

    def predict(self, payload):
        def forward_run(inputs):
            batches = get_batches_ex(inputs, self.batch_size, 4)
            logit_list = []
            attention_gradient_list = []
            for batch in batches:
                feed_dict = models.bert_util.bert_utils.batch2feed_dict_4_or_5_inputs(self.task, batch)
                logits, attention_gradient = self.sess.run([self.task.logits, self.attention_gradient],
                                                           feed_dict=feed_dict)
                logit_list.append(logits)
                attention_gradient_list.append(attention_gradient)
            return np.concatenate(logit_list), np.concatenate(attention_gradient_list)

        payload = [self.unpack_dict(e) for e in payload]
        logits, grads = forward_run(payload)
        return logits.tolist(), grads.tolist()


def get_attention_mask_gradient_predictor():
    save_path = os.path.join(output_path, "model", "runs", "mmd_Z")
    predictor = PredictorAttentionMaskGradient(2, 512)
    predictor.load_model(save_path)
    return predictor
