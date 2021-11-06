from typing import NamedTuple, List

import numpy as np

from bert_api.msmarco_tokenization import pretty_tokens


class QTypeInstance(NamedTuple):
    query: List[str]
    drop_query: List[str]
    doc: List[str]
    qtype_weights: np.array
    label: int

    def get_function_terms(self):
        return [t for t in self.query if t not in self.drop_query]

    def summary(self):
        return "[{}] {}".format(" ".join(self.get_function_terms()), pretty_tokens(self.drop_query, True))