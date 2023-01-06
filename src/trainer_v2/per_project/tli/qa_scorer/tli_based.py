from typing import List, Iterable, Callable, Dict, Tuple, Set

from trainer_v2.keras_server.name_short_cuts import get_pep_cache_client
from trainer_v2.per_project.tli.enum_subseq import enum_subseq_136
from trainer_v2.per_project.tli.tli_wrapper import TLIQDWrapper
from trainer_v2.per_project.tli.token_level_inference import nc_max_e_avg_reduce_then_norm, TokenLevelInference


def get_tli_based_models(name) -> Callable[[List[Tuple[str, str]]], List[float]]:
    if name == "tli2":
        nli_predict_fn = get_pep_cache_client()
        tli_module = TokenLevelInference(nli_predict_fn, enum_subseq_136)

        def reducer(s1, s2, tli_out) -> float:
            return nc_max_e_avg_reduce_then_norm(tli_out)[0]

        tli_wrapper = TLIQDWrapper(tli_module, reducer)
        return tli_wrapper.batch_predict
    else:
        assert False