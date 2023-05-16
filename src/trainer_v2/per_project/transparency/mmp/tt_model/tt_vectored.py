from typing import Dict

from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.per_project.transparency.mmp.tt_model.encoders import TermVector
from tf_util.lib.tf_funcs import find_layer
from trainer_v2.per_project.transparency.mmp.tt_model.tt1 import define_inputs, ScoringLayer2, \
    get_tf_loss
from trainer_v2.per_project.transparency.mmp.tt_model.model_conf_defs import InputShapeConfigTT
import tensorflow as tf
from tensorflow import keras


def load_word_embedding(
        embedding: keras.layers.Embedding,
        ckpt_path: str
):
    weight = embedding.weights[0]
    target_name = "bert/embeddings/word_embeddings"
    ckpt_reader = tf.train.load_checkpoint(ckpt_path)
    param_values = keras.backend.batch_get_value([weight])
    param_value = param_values[0]
    ckpt_value = ckpt_reader.get_tensor(target_name)
    weight_value_tuples = [(weight, ckpt_value)]

    if param_value.shape != ckpt_value.shape:
        raise ValueError("Shape {} does not match {}".format(param_value.shape, ckpt_value.shape))

    keras.backend.batch_set_value(weight_value_tuples)


class TTVectorBasedTrainNetwork:
    def __init__(
            self,
            bert_params,
            config: InputShapeConfigTT,
            loss_fn,
            scoring_layer_factory,
            alpha=0.1
    ):
        q_term_encoder = TermVector(bert_params, "query")
        d_term_encoder = TermVector(bert_params, "doc")
        self.q_term_encoder = q_term_encoder
        self.d_term_encoder = d_term_encoder

        window_len = config.max_subword_per_word
        num_window = config.max_terms
        bow_reps, inputs = define_inputs(num_window, window_len)

        batch_size, _, _ = get_shape_list2(bow_reps['q']['input_ids'])
        q_rep = q_term_encoder(bow_reps['q']['input_ids'])
        d1_rep = d_term_encoder(bow_reps['d1']['input_ids'])
        d2_rep = d_term_encoder(bow_reps['d2']['input_ids'])
        get_doc_score = scoring_layer_factory()
        s1, ex_tf1 = get_doc_score([q_rep, d1_rep, bow_reps['q'], bow_reps['d1']])
        s2, ex_tf2 = get_doc_score([q_rep, d2_rep, bow_reps['q'], bow_reps['d2']])
        pairwise_loss = loss_fn(s1, s2)
        # apply loss on magnitude of expanded terms
        loss_tf1 = get_tf_loss(ex_tf1)
        loss_tf2 = get_tf_loss(ex_tf2)
        verbosity_loss = tf.maximum(loss_tf1 + loss_tf2, 1)
        self.pairwise_loss = pairwise_loss
        loss = pairwise_loss + alpha * verbosity_loss
        output_d = {
            's1': s1,
            's2': s2,
            'pairwise_loss': pairwise_loss,
            'verbosity_loss': verbosity_loss,
            'loss': loss
        }
        model = keras.Model(inputs=inputs, outputs=output_d, name="bow_translation_table")
        self.loss = loss
        self.verbosity_loss = verbosity_loss
        self.model: keras.Model = model

    def get_tensors_to_log(self) -> Dict[str, tf.Tensor]:
        return {
            'verbosity_loss': self.verbosity_loss,
            'loss_cont': self.pairwise_loss
        }

    def init_embeddings(self, ckpt):
        load_word_embedding(self.d_term_encoder.embeddings_layer, ckpt)
        load_word_embedding(self.q_term_encoder.embeddings_layer, ckpt)


class TTVectorBasedTrainNetworkTrainable:
    def __init__(
            self, bert_params,
            config: InputShapeConfigTT,
            loss_fn,
            scoring_layer_factory,
    ):
        q_term_encoder = TermVector(bert_params, "query", True)
        d_term_encoder = TermVector(bert_params, "doc", True)
        self.q_term_encoder = q_term_encoder
        self.d_term_encoder = d_term_encoder

        window_len = config.max_subword_per_word
        num_window = config.max_terms
        bow_reps, inputs = define_inputs(num_window, window_len)

        batch_size, _, _ = get_shape_list2(bow_reps['q']['input_ids'])
        q_rep = q_term_encoder(bow_reps['q']['input_ids'])
        d1_rep = d_term_encoder(bow_reps['d1']['input_ids'])
        d2_rep = d_term_encoder(bow_reps['d2']['input_ids'])
        get_doc_score = scoring_layer_factory()
        s1, ex_tf1 = get_doc_score([q_rep, d1_rep, bow_reps['q'], bow_reps['d1']])
        s2, ex_tf2 = get_doc_score([q_rep, d2_rep, bow_reps['q'], bow_reps['d2']])
        pairwise_loss = loss_fn(s1, s2)
        # apply loss on magnitude of expanded terms
        loss_tf1 = get_tf_loss(ex_tf1)
        loss_tf2 = get_tf_loss(ex_tf2)
        verbosity_loss = tf.maximum(loss_tf1 + loss_tf2, 1)
        self.pairwise_loss = pairwise_loss
        alpha = 0.1
        loss = pairwise_loss + alpha * verbosity_loss
        output_d = {
            's1': s1,
            's2': s2,
            'pairwise_loss': pairwise_loss,
            'verbosity_loss': verbosity_loss,
            'loss': loss
        }
        model = keras.Model(inputs=inputs, outputs=output_d, name="bow_translation_table")
        self.loss = loss
        self.verbosity_loss = verbosity_loss
        self.model: keras.Model = model

    def get_tensors_to_log(self) -> Dict[str, tf.Tensor]:
        return {
            'verbosity_loss': self.verbosity_loss,
            'loss_cont': self.pairwise_loss
        }

    def init_embeddings(self, ckpt):
        load_word_embedding(self.d_term_encoder.embeddings_layer, ckpt)
        load_word_embedding(self.q_term_encoder.embeddings_layer, ckpt)

