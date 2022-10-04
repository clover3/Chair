from typing import NamedTuple


class TextPairProblem(NamedTuple):
    problem_id: str
    text1: str
    text2: str

    def get_problem_id(self):
        return self.problem_id
