import json
from typing import List, Iterable, Callable, Dict, Tuple, Set
from cpath import output_path
from misc_lib import path_join

from bert_api import SegmentedText
from bert_api.segmented_instance.segmented_text import token_list_to_segmented_text, merge_subtoken_level_scores
from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.scientsbank.parse_fns import SplitSpec, load_scientsbank_split
from dataset_specific.scientsbank.pte_data_types import ReferenceAnswer, Facet, Question
from dataset_specific.scientsbank.pte_solver_if import PTESolverIF
from misc_lib import path_join, average
from trainer_v2.per_project.tli.pte.runner.print_problems import enum_sa_ra
from trainer_v2.per_project.tli.pte.solver_adapter import get_score_for_facet


class SLRSolverForPTE(PTESolverIF):
    def __init__(self,
                 token_score_d: Dict[str, List[float]],
                 name: str
                 ):
        self.token_score_d: Dict[str, List[float]] = token_score_d
        self.name = name
        self.tokenizer = get_tokenizer()

    def get_name(self):
        return self.name

    def solve(self,
              reference_answer: ReferenceAnswer,
              student_answer: str,
              facet: Facet) -> float:
        key = "{}_{}".format(student_answer, reference_answer.text)
        # Find index of facets and
        token_neutral_scores = self.token_score_d[key]
        token_entail_scores = [1 - t for t in token_neutral_scores]
        h_tokens: List[str] = [t.text for t in reference_answer.tokens]

        t2: SegmentedText = token_list_to_segmented_text(self.tokenizer, h_tokens)
        if len(t2.tokens_ids) != len(token_entail_scores):
            tokens = self.tokenizer.convert_ids_to_tokens(t2.tokens_ids)
            can_idx = tokens.index("can")
            if tokens[can_idx+1] == "not":
                print("Manual fix")
                token_entail_scores = token_entail_scores[:can_idx] + [token_entail_scores[can_idx]] + token_entail_scores[can_idx:]


        assert len(t2.tokens_ids) == len(token_entail_scores)
        word_scores: List[float] = merge_subtoken_level_scores(average, token_entail_scores, t2)
        output_score = get_score_for_facet(reference_answer, facet, word_scores)
        return output_score


def read_scores_for_split(split_spec: SplitSpec) -> Dict[str, List[float]]:
    label = "mismatch"
    basename = split_spec.get_save_name() + ".txt"
    score_path = path_join(output_path, "pte_scientsbank",
                           "slr", f"{label}_plain_{basename}.txt")
    f = open(score_path, "r")
    scores: List[List[float]] = [json.loads(line) for line in f]
    questions: List[Question] = load_scientsbank_split(split_spec)
    score_d = {}
    for (prem, hypo), score in zip(enum_sa_ra(questions), scores):
        key = f"{prem}_{hypo}"
        score_d[key] = score
    return score_d



