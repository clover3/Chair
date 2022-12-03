from typing import List

from trainer_v2.keras_server.name_short_cuts import NLIPredictorSig


def get_em_base_nli() -> NLIPredictorSig:
    def tokenize_normalize(chunk):
        tokens = chunk.lower().split()
        return tokens

    def em_based_nli(t1: str, t2: str) -> List[float]:
        tokens1 = tokenize_normalize(t1)
        tokens2 = tokenize_normalize(t2)

        entail = True
        for t in tokens2:
            if t in tokens1:
                pass
            else:
                entail = False

        if entail:
            return [1, 0, 0]
        else:
            return [0, 1, 0]

    def func(pair_items) -> List[List[float]]:
        return [em_based_nli(t1, t2) for t1, t2 in pair_items]
    return func


