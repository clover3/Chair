from contradiction.medical_claims.cont_classification.defs import ContProblem, ContClassificationSolverNB
from misc_lib import pick1


class Majority(ContClassificationSolverNB):
    def solve(self, problem: ContProblem) -> int:
        return 0


class RandomClassifier(ContClassificationSolverNB):
    def solve(self, problem: ContProblem) -> int:
        return pick1([0, 1])
