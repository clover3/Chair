import time
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF
from data_generator.NLI.enlidef import NEUTRAL
from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.seg_encoder_common import encode_two_segments
from list_lib import right, left
from misc_lib import Averager, ceil_divide, tprint
from tlm.data_gen.base import get_basic_input_feature_as_list
from trainer_v2.custom_loop.modeling_common.tf_helper import distribute_dataset
from trainer_v2.custom_loop.neural_network_def.siamese import ModelConfig200_200
from trainer_v2.custom_loop.per_task.nli_ts_util import load_local_decision_nli_model

Probs = List[float]


# TODO try multiple window size
# TODO make speed faster by batch execution

class LocalDecisionNLICore:
    def __init__(self, model_path, strategy):
        tprint("Loading model...")
        self.predictor = load_local_decision_nli_model(model_path)
        tprint("Done")
        self.strategy = strategy

    def predict(self, input_list):
        batch_size = 16
        while len(input_list) % batch_size:
            input_list.append(input_list[-1])
        dataset = tf.data.Dataset.from_tensor_slices(input_list)
        strategy = self.strategy

        def reform(row):
            x = row[0], row[1], row[2], row[3]
            return x,

        dataset = dataset.map(reform)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        maybe_step = ceil_divide(len(input_list), batch_size)
        dataset = distribute_dataset(strategy, dataset)
        model = self.predictor
        l_decision, g_decision = model.predict(dataset, steps=maybe_step)
        return l_decision


class CachingAdapter:
    def __init__(self, inner_fn, data_shape):
        self.inner_fn = inner_fn
        self.key_n_payload = []
        self.mapping = {}
        self.data_shape = data_shape

    def register_payloads(self, input_list):
        for e in input_list:
            key = str(e)
            self.key_n_payload.append((key, e))

        return np.zeros([len(input_list)] + self.data_shape)

    def batch_predict(self):
        payload = right(self.key_n_payload)
        key_list = left(self.key_n_payload)
        results = self.inner_fn(payload)
        for key, result in zip(key_list, results):
            self.mapping[key] = result

    def predict(self, input_list):
        output = []
        for e in input_list:
            key = str(e)
            output.append(self.mapping[key])
        return output


class NLITSSolver(TokenScoringSolverIF):
    def __init__(self, predict_fn, max_seq_length1, max_seq_length2, target_label=NEUTRAL):
        self.tokenizer = get_tokenizer()
        self.max_seq_length1 = max_seq_length1
        self.segment_len = int(max_seq_length2 / 2)
        # self.predictor = load_local_decision_nli_model(model_path)
        self.target_label = target_label
        self.elapsed_all = 0
        self.elapsed_tf = 0
        self.predict_wrap = predict_fn

    def solve(self, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        st = time.time()
        t2_scores = self.solve_for_second(text1_tokens, text2_tokens)
        t1_scores = self.solve_for_second(text2_tokens, text1_tokens)
        elapsed = time.time() - st
        self.elapsed_all += elapsed
        return t1_scores, t2_scores

    def sb_tokenize(self, tokens):
        output = []
        for t in tokens:
            output.extend(self.tokenizer.tokenize(t))
        return output

    def encode_prem(self, tokens):
        tokens = self.sb_tokenize(tokens)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        input_ids, input_mask, segment_ids = get_basic_input_feature_as_list(self.tokenizer, self.max_seq_length1,
                                                                             tokens, segment_ids)
        return input_ids, segment_ids

    def solve_for_second(self, text1_tokens: List[str], text2_tokens: List[str]) -> List[float]:
        prem_inputs = self.encode_prem(text1_tokens)
        records: List[Tuple[Probs, int, int]] = self.get_local_decisions(prem_inputs, text2_tokens, 3)
        scores = self.accumulate_record_scores(records, text2_tokens)
        return scores

    def accumulate_record_scores(self, records, text2_tokens) -> List[float]:
        scores_building = [Averager() for _ in text2_tokens]
        for probs, st, ed in records:
            s = probs[self.target_label]
            for i in range(st, ed):
                if i < len(scores_building):
                    scores_building[i].append(s)
        scores = [s.get_average() for s in scores_building]
        return scores

    def get_local_decisions(self, prem_inputs, text2_tokens, window_size):
        p_x0, p_x1 = prem_inputs
        records: List = []
        input_list = []
        st_ed_list = []
        for hypo_inputs in self.enum_hypo_tuples(text2_tokens, window_size):
            h_x0, h_x1, st, ed = hypo_inputs
            st_ed_list.append((st, ed))
            x = p_x0, p_x1, h_x0, h_x1
            input_list.append(tuple(x))

        real_input_len = len(input_list)

        st = time.time()
        l_decision = self.predict_wrap(input_list)
        elapsed = time.time() - st
        self.elapsed_tf += elapsed

        for idx in range(real_input_len):
            second_l_decision = l_decision[idx][1]
            assert len(second_l_decision) == 3
            st, ed = st_ed_list[idx]
            records.append((second_l_decision, st, ed))
        return records

    def enum_hypo_tuples(self, tokens, window_size):
        space_tokenized_tokens = tokens
        st = 0
        sb_tokenize = self.sb_tokenize
        step_size = 1

        while st < len(space_tokenized_tokens):
            ed = st + window_size
            first_a = space_tokenized_tokens[:st]
            second = space_tokenized_tokens[st:ed]
            first_b = space_tokenized_tokens[ed:]
            first = sb_tokenize(first_a) + ["[MASK]"] + sb_tokenize(first_b)
            second = sb_tokenize(second)

            all_input_ids, all_input_mask, all_segment_ids = \
                encode_two_segments(self.tokenizer, self.segment_len, first, second)
            yield all_input_ids, all_segment_ids, st, ed
            st += step_size


def get_nli_ts34_solver(model_path, strategy, target_label):
    model_config = ModelConfig200_200()
    ld_core = LocalDecisionNLICore(model_path, strategy)
    return NLITSSolver(ld_core, model_config.max_seq_length1, model_config.max_seq_length2, target_label)