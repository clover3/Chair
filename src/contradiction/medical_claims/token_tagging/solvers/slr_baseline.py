import json
from typing import List, Iterable, Callable, Dict, Tuple, Set

# Move to Chair
from bert_api.segmented_instance.segmented_text import token_list_to_segmented_text, SegmentedText, \
    merge_subtoken_level_scores
from contradiction.medical_claims.token_tagging.batch_solver_common import BatchTokenScoringSolverIF
from contradiction.medical_claims.token_tagging.problem_loader import AlamriProblem
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import average


def two_token_list_to_text(tokens1, tokens2):
    return " ".join(tokens1) + "," + " ".join(tokens2)


def solve_problems(
        problems: List[AlamriProblem],
        token_score_list: List[Tuple[List[float], List[float]]]):
    tokenizer = get_tokenizer()
    assert len(problems) == len(token_score_list)

    answer_d = {}
    for p, score_pair in zip(problems, token_score_list):
        text1_tokens = p.text1.split()
        text2_tokens = p.text2.split()
        t1: SegmentedText = token_list_to_segmented_text(tokenizer, text1_tokens)
        t2: SegmentedText = token_list_to_segmented_text(tokenizer, text2_tokens)
        sb_scores1, sb_scores2 = score_pair

        assert len(t1.tokens_ids) == len(sb_scores1)
        assert len(t2.tokens_ids) == len(sb_scores2)

        word_scores1: List[float] = merge_subtoken_level_scores(average, sb_scores1, t1)
        word_scores2: List[float] = merge_subtoken_level_scores(average, sb_scores2, t2)
        key: str = two_token_list_to_text(text1_tokens, text2_tokens)

        answer_d[key] = (word_scores1, word_scores2)
    return answer_d


class TokenSolverWithAnswerD(BatchTokenScoringSolverIF):
    def __init__(self, answer_d):
        self.answer_d = answer_d

    def solve(self, payload: List[Tuple[List[str], List[str]]])\
            -> List[Tuple[List[float], List[float]]]:
        output_list = []
        for tokens1, tokens2 in payload:
            key = two_token_list_to_text(tokens1, tokens2)
            output = self.answer_d[key]
            output_list.append(output)
        return output_list


def get_snli_logic_solver(problems, score_path) -> TokenSolverWithAnswerD:
    f = open(score_path, "r")
    score_pairs = [json.loads(line) for line in f]
    answer_d = solve_problems(problems, score_pairs)
    return TokenSolverWithAnswerD(answer_d)


