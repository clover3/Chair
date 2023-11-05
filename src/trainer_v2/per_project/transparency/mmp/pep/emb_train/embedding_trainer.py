from typing import Iterable, Tuple

import numpy as np
import tensorflow as tf

from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.hf_encode_helper import combine_with_sep_cls_inner
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.definitions import ModelConfigType
from trainer_v2.custom_loop.neural_network_def.ts_emb_backprop import TSEmbBackprop, EmbTrainIF
from trainer_v2.custom_loop.per_task.pairwise_trainer import TrainerForLossReturningModel
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.per_project.cip.tfrecord_gen import pad_to_length
from trainer_v2.per_project.transparency.mmp.pep.pep_rerank_w_es import InputIdsSegmentIds


def format_to_dataset(max_seq_length, is_for_training, sequence_list, batch_size=None):
    SpecI = tf.TensorSpec([max_seq_length], dtype=tf.int32)
    sig = (SpecI, SpecI)

    def generator() -> Iterable[Tuple[InputIdsSegmentIds]]:
        for sequence in sequence_list:
            input_ids, segment_ids = sequence
            input_ids = pad_to_length(input_ids, max_seq_length)
            segment_ids = pad_to_length(segment_ids, max_seq_length)
            yield input_ids, segment_ids

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=sig)

    if batch_size is None:
        batch_size = len(sequence_list)

    dataset = dataset.batch(batch_size)
    if is_for_training:
        dataset = dataset.repeat()
    return dataset


class EmbeddingTrainer(TrainerForLossReturningModel):
    def __init__(
            self,
            run_config: RunConfig2,
            inner_model: EmbTrainIF,
            target_q_token: str,
            model_config: ModelConfigType
    ):
        self.emb_model: EmbTrainIF = inner_model
        super(TrainerForLossReturningModel, self).__init__(run_config, inner_model)
        self.neg_spe_emb_idx = -1
        self.baseline_loss = None
        self.tokenizer = get_tokenizer()
        self.target_q_token = target_q_token
        self.segment_len: int = model_config.max_seq_length // 2
        self.target_emb_last = None

    def do_init_checkpoint(self, init_checkpoint):
        pass

    def get_eval_object(self, eval_batches, strategy):
        self.eval_batches = eval_batches
        self.strategy = strategy
        return self

    def do_eval(self):
        word_emb_layer: tf.keras.layers.Embedding = self.emb_model.get_word_embedding_layer()
        target_q_token = self.target_q_token

        def get_row_of_embedding(emb_layer, i) -> np.array:
            embedding_weights = emb_layer.get_weights()[0]
            ith_embedding = embedding_weights[i]
            return ith_embedding

        target_emb = self.emb_model.get_target_rep(self.neg_spe_emb_idx)
        if self.target_emb_last is not None:
            vector_change = np.sum(np.abs(self.target_emb_last - target_emb))
        else:
            vector_change = 0
        self.target_emb_last = target_emb
        bias_to_added = get_row_of_embedding(word_emb_layer, 0)
        effective_emb = target_emb + bias_to_added
        model = self.emb_model.get_keras_model()

        if self.baseline_loss is None:
            with self.strategy.scope():
                seq = self.get_sequence(target_q_token, target_q_token)
                baseline_dataset = format_to_dataset(self.segment_len, False, [seq])

            for batch in baseline_dataset:
                pred, loss = model.predict_on_batch(batch)
                c_log.info(f"Baseline loss={loss}, pred={pred[0][0]}")
                self.baseline_loss = loss
                break

        all_word_emb = word_emb_layer.get_weights()[0]
        all_word_emb_norm = tf.nn.l2_normalize(all_word_emb, axis=1)
        sim = tf.reduce_sum(all_word_emb_norm * effective_emb, axis=1)
        sim = sim.numpy()
        rank = np.argsort(sim)[::-1]
        k = 5
        top_tokens_ids = [rank[i] for i in range(k)]
        top_subword_terms = [self.tokenizer.inv_vocab[i] for i in top_tokens_ids]
        seq_list = [self.get_sequence(target_q_token, token) for token in top_subword_terms]
        eval_dataset: tf.data.Dataset = format_to_dataset(self.segment_len, False, seq_list)

        def two_to_one(x1, x2):
            return (x1, x2),

        eval_dataset = eval_dataset.map(two_to_one)
        pred, loss = model.predict(eval_dataset)

        for i in range(k):
            token_id: int = rank[i].tolist()
            sim_val = sim[token_id]
            token = self.tokenizer.inv_vocab[token_id]
            pred_val = pred[i][0]
            print(f"Rank {i}: {token} {sim_val:.2f}  {pred_val:.2f}")

        metrics = {"vector change": vector_change}
        dummy_loss = tf.constant(0.0)
        return dummy_loss, metrics

    def build_dataset_spe1(self, is_for_training):
        max_seq_length = self.segment_len

        def get_sequence():
            special_token = "[unused1]"
            tokens1 = ["[MASK]"] * 2 + [self.target_q_token] + ["[MASK]"] * 4
            tokens2 = ["[MASK]"] * 4 + [special_token] + ["[MASK]"] * 8
            tokens, segment_ids = combine_with_sep_cls_inner(max_seq_length, tokens1, tokens2)
            idx = tokens.index(special_token)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids[idx] = self.neg_spe_emb_idx
            return input_ids, segment_ids

        return format_to_dataset(self.segment_len, is_for_training, [get_sequence()])

    def get_sequence(self, q_token_id, d_token_id):
        max_seq_length = self.segment_len
        tokens1 = ["[MASK]"] * 2 + [q_token_id] + ["[MASK]"] * 4
        tokens2 = ["[MASK]"] * 4 + [d_token_id] + ["[MASK]"] * 8
        tokens, segment_ids = combine_with_sep_cls_inner(max_seq_length, tokens1, tokens2)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return input_ids, segment_ids


#  Get attention key/query vector for each terms,
#   IF they have similar key/query attention vectors they will probably be related.
#   Note that some key/query may not be relevant, as they do not contributes to the decisions.
#
#   If we need to compute higher layers, maybe that requires too much computations
#   Option 1. Compute first layer
#   Option 2. Put first layer vectors to other layers key/query values
#   Option 3. Compute PEP for all d_term pairs
#   Option 4. Multiply to get scores and do weighted sum
#   Option 5.
# We want to get a vectors (H, ) size which can be multiplied to (V, H) vector to get scores
