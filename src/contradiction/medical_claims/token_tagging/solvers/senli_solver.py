from typing import List, Tuple

from bert_api.segmented_instance.segmented_text import token_list_to_segmented_text, SegmentedText, \
    merge_subtoken_level_scores
from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF, \
    TokenScoringSolverIFOneWay
from cpath import pjoin, data_path, get_model_full_path
from data_generator.NLI.enlidef import NEUTRAL
from data_generator.bert_input_splitter import split_p_h_with_input_ids
from data_generator.shared_setting import BertNLI
from data_generator.tokenizer_wo_tf import get_tokenizer, EncoderUnitPlain
from explain.nli_ex_predictor import NLIExPredictor
from misc_lib import average
from models.transformer import hyperparams


class SENLISolver(TokenScoringSolverIFOneWay):
    def __init__(self, predictor, target_label=NEUTRAL, max_seq_length=300):
        self.predictor: NLIExPredictor = predictor
        self.tokenizer = get_tokenizer()
        self.target_label = target_label
        voca_path = pjoin(data_path, "bert_voca.txt")
        self.d_encoder = EncoderUnitPlain(max_seq_length, voca_path)

    def solve(self, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        t1: SegmentedText = token_list_to_segmented_text(self.tokenizer, text1_tokens)
        t2: SegmentedText = token_list_to_segmented_text(self.tokenizer, text2_tokens)
        t1_scores = self.solve_for_second(t2, t1)
        t2_scores = self.solve_for_second(t1, t2)
        return t1_scores, t2_scores

    def solve_for_second(self, t1: SegmentedText, t2: SegmentedText) -> List[float]:
        d = self.d_encoder.encode_inner(t1.tokens_ids, t2.tokens_ids)
        single_x = d["input_ids"], d["input_mask"], d["segment_ids"]
        items = [single_x]
        sout, ex_logits = self.predictor.predict_both_from_insts(self.target_label, items)
        all_scores = ex_logits[0]
        input_ids = d['input_ids']
        conf_p, conf_h = split_p_h_with_input_ids(all_scores, input_ids)
        seg2_scores = conf_h
        if len(conf_h) != len(t2.tokens_ids):
            print(len(conf_h), len(t2.tokens_ids))
        scores = merge_subtoken_level_scores(average, seg2_scores, t2)
        return scores

    def solve_one_way(self, text1_tokens: List[str], text2_tokens: List[str]):
        t1: SegmentedText = token_list_to_segmented_text(self.tokenizer, text1_tokens)
        t2: SegmentedText = token_list_to_segmented_text(self.tokenizer, text2_tokens)
        t2_scores = self.solve_for_second(t1, t2)
        return t2_scores



class SENLISolverAsym(TokenScoringSolverIF):
    def __init__(self, predictor, target_label=NEUTRAL, max_seq_length=300):
        self.predictor: NLIExPredictor = predictor
        self.tokenizer = get_tokenizer()
        self.target_label = target_label
        voca_path = pjoin(data_path, "bert_voca.txt")
        self.d_encoder = EncoderUnitPlain(max_seq_length, voca_path)

    def solve(self, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        t1: SegmentedText = token_list_to_segmented_text(self.tokenizer, text1_tokens)
        t2: SegmentedText = token_list_to_segmented_text(self.tokenizer, text2_tokens)
        d = self.d_encoder.encode_inner(t1.tokens_ids, t2.tokens_ids)
        single_x = d["input_ids"], d["input_mask"], d["segment_ids"]
        items = [single_x]
        sout, ex_logits = self.predictor.predict_both_from_insts(self.target_label, items)
        all_scores = ex_logits[0]
        input_ids = d['input_ids']
        seg1_scores, seg2_scores = split_p_h_with_input_ids(all_scores, input_ids)
        scores1 = merge_subtoken_level_scores(average, seg1_scores, t1)
        scores2 = merge_subtoken_level_scores(average, seg2_scores, t2)

        return scores1, scores2


def get_nli_ex_19():
    hp = hyperparams.HPSENLI3()
    nli_setting = BertNLI()
    model_path = get_model_full_path("nli_ex_19", )
    modeling_option = "co"
    predictor = NLIExPredictor(hp, nli_setting, model_path, modeling_option)
    return predictor


def get_senli_solver(target_label):
    return SENLISolver(get_nli_ex_19(), target_label)


def get_senli_solver_aym(target_label):
    return SENLISolverAsym(get_nli_ex_19(), target_label)


def main():
    get_nli_ex_19()


if __name__ == "__main__":
    main()