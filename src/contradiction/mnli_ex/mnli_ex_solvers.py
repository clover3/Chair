import random
from typing import List, Tuple

from arg.clueweb12_B13_termstat import ClueIDF
from attribution.attrib_types import TokenScores
from contradiction.mnli_ex.load_mnli_ex_data import MNLIExEntry
from contradiction.mnli_ex.solver_common import MNLIExSolver
from data_generator.tokenizer_wo_tf import get_tokenizer


class IdfScorer2(MNLIExSolver):
    def __init__(self):
        self.idf_module = ClueIDF()

    def explain(self, data: List[MNLIExEntry], target_label) -> List[Tuple[TokenScores, TokenScores]]:
        if target_label != "mismatch":
            print("IdfScorer2 is implemented only for mismatch")
        explains: List[Tuple[TokenScores, TokenScores]] = []
        for entry in data:
            tokens1 = entry.premise.lower().split()
            tokens2 = entry.hypothesis.lower().split()

            def solve_for_second(tokens1, tokens2):
                raw_scores = []
                for t in tokens2:
                    idf = self.idf_module.get_weight(t)
                    if t in tokens1:
                        sign = -1
                    else:
                        sign = 1

                    s = sign * idf
                    raw_scores.append(s)
                return raw_scores_to_paired(raw_scores)

            scores_pair = solve_for_second(tokens2, tokens1), solve_for_second(tokens1, tokens2)
            explains.append(scores_pair)
        return explains


def raw_scores_to_paired(raw_scores: List[float]):
    scores = [(score, idx) for idx, score in enumerate(raw_scores)]
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores


class ExactMatchScorer(MNLIExSolver):
    def explain(self, data: List[MNLIExEntry], target_label) -> List[Tuple[TokenScores, TokenScores]]:
        if target_label != "mismatch":
            print("IdfScorer2 is implemented only for mismatch")
        explains: List[Tuple[TokenScores, TokenScores]] = []
        for entry in data:
            tokens1 = entry.premise.lower().split()
            tokens2 = entry.hypothesis.lower().split()

            def solve_for_second(tokens1, tokens2):
                raw_scores = []
                for t in tokens2:
                    if t in tokens1:
                        s = 0
                    else:
                        s = 1
                    raw_scores.append(s)
                return raw_scores_to_paired(raw_scores)

            scores_pair = solve_for_second(tokens2, tokens1), solve_for_second(tokens1, tokens2)
            explains.append(scores_pair)
        return explains


def get_random_score_from_text(t) -> TokenScores:
    scores = [(random.random(), idx) for idx, _ in enumerate(t.split())]
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores


class RandomScorer2(MNLIExSolver):
    def __init__(self):
        self.tokenizer = get_tokenizer()

    def explain(self, data: List[MNLIExEntry], target_label) -> List[Tuple[TokenScores, TokenScores]]:
        explains: List[Tuple[TokenScores, TokenScores]] = []
        for entry in data:
            scores_pair = get_random_score_from_text(entry.premise), get_random_score_from_text(entry.hypothesis)
            explains.append(scores_pair)
        return explains

