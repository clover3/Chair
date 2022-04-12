from typing import List

from data_generator.tokenizer_wo_tf import get_tokenizer
from bert_api.bert_masking_common import later_score_prob
from alignment.matrix_scorers.attn_based.perturbation_scorer import PerturbationScorer
from bert_api.task_clients.bert_masking_client import get_localhost_bert_mask_client
from alignment import MatrixScorerIF
from alignment.data_structure.related_eval_instance import RelatedEvalInstance
from tlm.qtype.partial_relevance.loader import load_mmde_problem


def get_scorer() -> MatrixScorerIF:
    predictor = get_localhost_bert_mask_client()
    max_seq_length = 512
    scorer = PerturbationScorer(predictor, max_seq_length, later_score_prob)
    return scorer


def main():
    dataset_name = "dev"
    problems: List[RelatedEvalInstance] = load_mmde_problem(dataset_name)
    tokenizer = get_tokenizer()
    for p in problems:
        tokens = tokenizer.convert_ids_to_tokens(p.seg_instance.text2.tokens_ids)
        print(tokens)
        for seg_idx in range(p.seg_instance.text2.get_seg_len()):
            print(list(p.seg_instance.text2.enum_token_idx_from_seg_idx(seg_idx)))
            for token_idx in p.seg_instance.text2.enum_token_idx_from_seg_idx(seg_idx):
                assert token_idx < 512


if __name__ == "__main__":
    main()
