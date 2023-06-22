from abc import ABC, abstractmethod
from typing import List, Iterable, Callable, Dict, Tuple, Set

from tqdm import tqdm

from dataset_specific.scientsbank.pte_data_types import Question, PTEPredictionPerQuestion, PTEPredictionPerFacet, \
    PTEPredictionPerStudentAnswer, Facet, ReferenceAnswer
from misc_lib import TEL, TimeEstimator
from trainer_v2.chair_logging import c_log


class PTESolverIF(ABC):
    @abstractmethod
    def solve(self,
              reference_answer: ReferenceAnswer,
              student_answer: str,
              facet: Facet) -> float:
        pass

    def get_name(self):
        return ""


def apply_solver(solver: PTESolverIF,
                 questions: List[Question],
                 threshold_fn=None
                 ) -> List[PTEPredictionPerQuestion]:
    n_student_answer = 0
    for q in questions:
        n_student_answer += len(q.student_answers)
    c_log.info("%d student answers", n_student_answer)
    ticker = TimeEstimator(n_student_answer)
    output = []
    for q in questions:
        per_question_answer = PTEPredictionPerQuestion(q.id)
        facet_d: Dict[str, Facet] = {}
        for facet in q.reference_answer.facets:
            facet_d[facet.id] = facet
        for sa in q.student_answers:
            per_student_answer = PTEPredictionPerStudentAnswer(sa.id)

            for fe in sa.facet_entailments:
                facet = facet_d[fe.facet_id]
                score = solver.solve(q.reference_answer, sa.answer_text, facet)
                if threshold_fn is not None:
                    pred = threshold_fn(score)
                else:
                    pred = score > 0.5
                facet_pred = PTEPredictionPerFacet(facet.id, score, pred)
                per_student_answer.facet_pred.append(facet_pred)

            per_question_answer.per_student_answer_list.append(per_student_answer)
            ticker.tick()
        output.append(per_question_answer)
    return output
    # return list(map(solve_question, questions))


class PTESolverAllTrue(PTESolverIF):
    def get_name(self):
        return "all_true"

    def solve(self,
              reference_answer: str,
              student_answer: str,
              facet: Tuple[str, str]) -> float:
        return 1


class PTESolverAllFalse(PTESolverIF):
    def get_name(self):
        return "all_false"

    def solve(self,
              reference_answer: str,
              student_answer: str,
              facet: Tuple[str, str]) -> float:
        return 0