from contradiction.medical_claims.cont_classification.defs import ContClassificationSolverNB, ContProblem
from trainer_v2.per_project.tli.bioclaim_qa.bm25_system import BM25BioClaim
from trainer_v2.per_project.tli.bioclaim_qa.eval_helper import get_bioclaim_retrieval_corpus
from list_lib import right


class BM25Classifier(ContClassificationSolverNB):
    def __init__(self, split):
        _, claims = get_bioclaim_retrieval_corpus(split)
        inner = BM25BioClaim(right(claims))
        self.bm25 = inner.bm25
        self.threshold = 1

    def solve(self, problem: ContProblem) -> int:
        score = self.get_raw_score(problem)
        if score >= self.threshold:
            return 1
        else:
            return 0

    def get_raw_score(self, problem):
        s1 = self.bm25.score(problem.question, problem.claim1_text)
        s2 = self.bm25.score(problem.question, problem.claim2_text)
        s3 = self.bm25.score(problem.claim1_text, problem.claim2_text)
        score = s1 + s2 + s3
        return score


def get_bm25_solver(split) -> ContClassificationSolverNB:
    return BM25Classifier(split)

