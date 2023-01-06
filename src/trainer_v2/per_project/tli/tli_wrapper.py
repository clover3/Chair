from trainer_v2.per_project.tli.token_level_inference import TokenLevelInference, Numpy2D
from typing import List, Iterable, Callable, Dict, Tuple, Set, Any, TypeVar

T = TypeVar("T")

# Because q should be hypothesis, it swaps them.

class TLIQDWrapper:
    def __init__(
            self,
            tli: TokenLevelInference,
            reducer: Callable[[str, str, Numpy2D], T],
    ):
        self.tli = tli
        self.reducer = reducer

    def batch_predict(self, q_d_pairs: List[Tuple[str, str]]) -> List[T]:
        d_q_pairs = [(d, q) for q, d in q_d_pairs]
        tli_d = self.tli.do_batch_return_dict(d_q_pairs)
        output = []
        for d, q in d_q_pairs:
            t = self.reducer(d, q, tli_d[d, q])
            output.append(t)
        return output




