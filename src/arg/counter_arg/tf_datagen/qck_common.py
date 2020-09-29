from arg.counter_arg.header import ArguDataPoint, Passage
from arg.qck.decl import QCKQuery, QCKCandidate


def passage_to_candidate(candidate: Passage) -> QCKCandidate:
    return QCKCandidate(candidate.id.id, candidate.text)


def problem_to_qck_query(problem: ArguDataPoint) -> QCKQuery:
    return QCKQuery(problem.text1.id.id, problem.text1.text)