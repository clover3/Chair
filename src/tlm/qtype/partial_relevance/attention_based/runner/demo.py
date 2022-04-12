from typing import List

from data_generator.tokenizer_wo_tf import get_tokenizer, ids_to_text
from tlm.qtype.partial_relevance.attention_based.attention_mask_eval import EvalPerQSeg
from bert_api.bert_masking_common import dist_l2
from alignment.matrix_scorers.attn_based.perturbation_scorer import PerturbationScorer
from bert_api.task_clients.bert_masking_client import get_localhost_bert_mask_client
from alignment.data_structure.related_eval_instance import RelatedEvalInstance
from tlm.qtype.partial_relevance.loader import load_mmde_problem


def main():
    problems: List[RelatedEvalInstance] = load_mmde_problem("dev_word")
    problems = problems[:2]
    predictor = get_localhost_bert_mask_client()
    max_seq_length = 512
    scorer = PerturbationScorer(predictor, max_seq_length, dist_l2)
    eval_model = EvalPerQSeg(predictor, max_seq_length)
    tokenizer = get_tokenizer()
    for p in problems:
        inst = p.seg_instance
        query = ids_to_text(tokenizer, inst.text1.tokens_ids)
        doc = ids_to_text(tokenizer, inst.text2.tokens_ids)
        print(query)
        print(doc)
        res = scorer.eval_contribution(inst)
        eval_model.verbose_print(inst, res)


if __name__ == "__main__":
    main()