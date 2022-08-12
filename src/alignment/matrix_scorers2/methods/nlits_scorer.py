from collections import defaultdict
from typing import List, Dict, Tuple

from alignment.data_structure.batch_scorer_if import BASAIF
from alignment.data_structure.matrix_scorer_if import MatrixScorerIF2
from bert_api.segmented_instance.segmented_text import token_list_to_segmented_text, SegmentedText
from contradiction.medical_claims.token_tagging.batch_solver_common import NeuralOutput
from contradiction.medical_claims.token_tagging.solvers.nlits_batch_solver import ESTwoPiece
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import average, group_by, two_digit_float
from trainer_v2.custom_loop.demo.demo_common import enum_hypo_token_tuple_from_tokens, EncodedSegmentIF
from trainer_v2.custom_loop.per_task.nli_ts_helper import get_local_decision_nlits_core
from trainer_v2.custom_loop.per_task.nli_ts_util import LocalDecisionNLICore
from trainer_v2.custom_loop.run_config2 import RunConfig2


class EncodedSegment(EncodedSegmentIF):
    def __init__(self, input_x,
                 p_drop,
                 st, ed):
        self.input_x = input_x
        self.p_drop = p_drop
        self.st = st
        self.ed = ed

    def get_input(self):
        return self.input_x


class NLITSAdapter(BASAIF):
    def __init__(self,
                 nlits: LocalDecisionNLICore,
                 ):
        self.nlits: LocalDecisionNLICore = nlits
        self.tokenizer = get_tokenizer()

    def enum_child(self, text1_tokens, text2_tokens):
        t1: SegmentedText = token_list_to_segmented_text(self.tokenizer, text1_tokens)

        es_list = []
        for h_first, h_second, st, ed in enum_hypo_token_tuple_from_tokens(self.tokenizer, text2_tokens, 1):
            x = self.nlits.encode_fn(self.tokenizer.convert_ids_to_tokens(t1.tokens_ids), h_first, h_second)
            es = EncodedSegment(x, -1, st, ed)
            es_list.append(es)
            for i in t1.enum_seg_idx():
                t1_sub = t1.get_dropped_text([i])
                p_tokens = self.tokenizer.convert_ids_to_tokens(t1_sub.tokens_ids)
                x = self.nlits.encode_fn(p_tokens, h_first, h_second)
                es = EncodedSegment(x, i, st, ed)
                es_list.append(es)
        return es_list

    def neural_worker(self, items: List[ESTwoPiece]) -> List[Tuple[NeuralOutput, ESTwoPiece]]:
        l_decisions = self.nlits.predict_es(items)
        return list(zip(l_decisions, items))

    def reduce(self, t1, t2, item: List[Tuple[NeuralOutput, EncodedSegment]]) -> List[List[float]]:
        def group_key(no_es: Tuple[NeuralOutput, EncodedSegment]) -> Tuple:
            no, es = no_es
            return es.st, es.ed

        grouped: Dict[Tuple, List[Tuple[NeuralOutput, EncodedSegment]]] = group_by(item, group_key)
        score_d: Dict[Tuple, List[float]] = defaultdict(list)
        # no: Neutral Output
        # es: Encoded segment
        print("t1:", t1)
        print("t2:", t2)
        for (st, ed), no_es_list in grouped.items():
            no_base, es_base = no_es_list[0]
            es_base: EncodedSegment = es_base
            assert es_base.p_drop == -1
            def get_entail(l):
                return l[0]
            before = get_entail(no_base)
            probs_str = ",".join(map(two_digit_float, no_base))
            print("{} t2={}".format(probs_str, " ".join(t2[st:ed])))
            for no, es in no_es_list[1:]:
                p_tokens = [t for idx, t in enumerate(t1) if es.p_drop != idx]
                after = get_entail(no)
                probs_str = ",".join(map(two_digit_float, no))
                print("{0}\t{1}\t(- {2})".format(probs_str, " ".join(p_tokens), t1[es.p_drop]))
                change = before - after
                for j in range(st, ed):
                    key = es.p_drop, j
                    score_d[key].append(change)

        score_array = []
        for i1 in range(len(t1)):
            row = []
            for i2 in range(len(t2)):
                change_list = score_d[i1, i2]
                if not change_list:
                    raise Exception(i1, i2)
                row.append(average(change_list))
            score_array.append(row)
        return score_array


class NLITSSolver(MatrixScorerIF2):
    def __init__(self, nlits_core):
        self.nlits_adapter = NLITSAdapter(nlits_core)

        self.tokenizer = get_tokenizer()

    def solve(self, tokens1: List[str], tokens2: List[str]) -> List[List[float]]:
        payloads = self.nlits_adapter.enum_child(tokens1, tokens2)
        n_output = self.nlits_adapter.neural_worker(payloads)
        mat_scores = self.nlits_adapter.reduce(tokens1, tokens2, n_output)
        return mat_scores


def get_nlits_2dsolver(run_config: RunConfig2, encoder_name: str) -> MatrixScorerIF2:
    nlits = get_local_decision_nlits_core(run_config, encoder_name)
    return NLITSSolver(nlits)


