from typing import NamedTuple


class ContProblem(NamedTuple):
    question: str
    claim1_text: str
    claim2_text: str
    label: int

    def signature(self):
        return "{}_{}_{}".format(self.question, self.claim1_text, self.claim2_text)

    @classmethod
    def from_json(cls, j):
        return ContProblem(j['question'], j['claim1_text'], j['claim2_text'], j['label'])