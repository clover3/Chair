import numpy as np
import numpy as np
import tensorflow as tf

import models.bert_util.bert_utils
from cpath import pjoin, data_path
from data_generator.bert_input_splitter import get_sep_loc
from data_generator.tokenizer_wo_tf import EncoderUnitPlain
from explain.bert_components.cmd_nli import ModelConfig
from models.transformer import hyperparams
from models.transformer.nli_base import transformer_nli_pooled_embedding_in
from tf_v2_support import disable_eager_execution
from trainer.np_modules import get_batches_ex
from trainer.tf_train_module_v2 import init_session


class GradientPredictorCore:
    def __init__(self, num_classes):
        disable_eager_execution()
        self.voca_size = 30522
        hp = hyperparams.HPFAD()
        self.task = transformer_nli_pooled_embedding_in(hp, self.voca_size, False)
        self.sess = init_session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.batch_size = 32
        self.model_loaded = False

        probs = tf.nn.softmax(self.task.logits, axis=1)
        logits = [probs[:, class_i] for class_i in range(num_classes)]
        self.gradient_to_embedding = tf.gradients(logits, self.task.encoded_embedding_in)
        # List [batch_size, seq_length, seq_length]

    def predict(self, payload):
        def forward_run(triple_list):
            batches = get_batches_ex(triple_list, self.batch_size, 3)
            logit_list = []
            gradient_list = []
            for batch in batches:
                feed_dict = models.bert_util.bert_utils.batch2feed_dict_4_or_5_inputs(self.task, batch)
                logits, gradients = self.sess.run([self.task.logits, self.gradient_to_embedding],
                                                           feed_dict=feed_dict)
                logit_list.append(logits)
                gradient_list.append(gradients)
            return np.concatenate(logit_list), np.concatenate(gradient_list)

        logits, grads = forward_run(payload)
        return logits, grads

    def load_model(self, save_path):
        self.loader = tf.compat.v1.train.Saver(max_to_keep=1)
        self.loader.restore(self.sess, save_path)
        self.model_loaded = True


class GradientPredictor:
    def __init__(self):
        model_config = ModelConfig()
        voca_path = pjoin(data_path, "bert_voca.txt")
        self.d_encoder = EncoderUnitPlain(model_config.max_seq_length, voca_path)
        self.core = GradientPredictorCore(model_config.num_classes)

    def predict(self, p_tokens_id, h_tokens_id):
        d = self.d_encoder.encode_inner(p_tokens_id, h_tokens_id)
        single_x = d["input_ids"], d["input_mask"], d["segment_ids"]
        logits, gradients = self.core.predict([single_x])
        logits = logits[0]
        gradients = gradients[0]

        sep_idx1, sep_idx2 = get_sep_loc(d["input_ids"])


