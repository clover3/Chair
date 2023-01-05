from typing import List, Iterable, Callable, Dict, Tuple, Set

from trainer_v2.per_project.tli.token_level_inference import TokenLevelInference, Numpy1D
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.per_project.tli.enum_subseq import enum_subseq_136
from contradiction.medical_claims.token_tagging.batch_solver_common import BatchTokenScoringSolverIF, ECCInput, \
    ECCOutput


class NLITSDualBatchSolver(BatchTokenScoringSolverIF):
    def __init__(
            self,
            tli_module: TokenLevelInference,
            target_label,
    ):
        self.tli_module = tli_module
        self.target_label = target_label

    def solve(self, problems: List[ECCInput]) -> List[ECCOutput]:
        problems_text = []
        for t1, t2 in problems:
            text1 = " ".join(t1)
            text2 = " ".join(t2)
            problems_text.append((text1, text2))

        tli_payload: List[Tuple[str, str]] = []
        for text1, text2 in problems_text:
            tli_payload.append((text1, text2))
            tli_payload.append((text2, text1))

        tli_payload = list(set(tli_payload))
        pred_d = self.tli_module.do_batch_return_dict(tli_payload)
        output = []
        for text1, text2 in problems_text:
            scores1 = pred_d[text1, text2][:, self.target_label]
            scores2 = pred_d[text2, text1][:, self.target_label]
            output.append((scores1, scores2))

        return output


def get_batch_solver_tli_based(run_config: RunConfig2, target_label: int):
    nli_predict_fn = get_nlits_direct(run_config)
    tli_module = TokenLevelInference(nli_predict_fn, enum_subseq_136)
    solver = NLITSDualBatchSolver(tli_module, target_label)
    return solver
