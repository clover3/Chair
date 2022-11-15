import abc

from alignment.ists_eval.chunked_eval import ISTSChunkedSolver, ISTSChunkedSolverNB
from dataset_specific.ists.parse import iSTSProblemWChunk, AlignmentPrediction, ALIGN_EQUI, ALIGN_SPE1, ALIGN_SPE2, \
    AlignmentLabelUnit, ALIGN_SIMI, ALIGN_NOALI
from typing import List, TypeVar, Tuple
from list_lib import lmap, foreach
from misc_lib import timed_lmap
from trainer.promise import EnumSubJobInterface, MyFuture, PromiseKeeper
from trainer.promise_ex import PromiseKeeperOnFuture
from trainer_v2.keras_server.name_short_cuts import NLIPredictorSig



# Option 1: Work on aligned outputs of Exact match or word2vec
class NLIDrivenSolver(ISTSChunkedSolver):
    def __init__(self, predict_fn: NLIPredictorSig, base_solver: ISTSChunkedSolverNB):
        self.predict_fn = predict_fn
        self.base_solver = base_solver

    def batch_solve(self, problems: List[iSTSProblemWChunk]) -> List[AlignmentPrediction]:
        return timed_lmap(self.solve_one, problems)

    def solve_one_futuristic(self, problem: iSTSProblemWChunk) -> Tuple[str, MyFuture[List[AlignmentLabelUnit]]]:
        base_alignment: AlignmentPrediction = self.base_solver.solve_one(problem)
        problem_id, alignment_list = base_alignment
        def is_entailment(probs):
            return probs[0] > 0.5

        NLIPred = List[float]

        pk = PromiseKeeper(self.predict_fn)

        def future_map1(alignment: AlignmentLabelUnit)\
                -> MyFuture[Tuple[AlignmentLabelUnit, NLIPred, NLIPred]]:
            nli_p1_MyFuture: MyFuture[NLIPred] = pk.get_future((alignment.chunk_text1, alignment.chunk_text2))
            nli_p2_MyFuture: MyFuture[NLIPred] = pk.get_future((alignment.chunk_text2, alignment.chunk_text1))
            MyFuture_obj = alignment, nli_p1_MyFuture, nli_p2_MyFuture
            return MyFuture_obj

        pk2 = PromiseKeeper(NotImplemented)

        def future_map2(t: Tuple[AlignmentLabelUnit, MyFuture[NLIPred], MyFuture[NLIPred]]) ->\
                MyFuture[Tuple[AlignmentLabelUnit, NLIPred, NLIPred]]:
            return pk2.get_future(t)

        def solve_core(t: Tuple[AlignmentLabelUnit, NLIPred, NLIPred]) -> AlignmentLabelUnit:
            alignment, nli_pred1, nli_pred2 = t
            if is_entailment(nli_pred1) and is_entailment(nli_pred2):
                align_types = [ALIGN_EQUI]
            elif is_entailment(nli_pred1) and not is_entailment(nli_pred2):
                align_types = [ALIGN_SPE1]
            elif not is_entailment(nli_pred1) and is_entailment(nli_pred2):
                align_types = [ALIGN_SPE2]
            else:
                align_types = [ALIGN_SIMI]

            alignment_new = AlignmentLabelUnit(
                alignment.chunk_token_id1,
                alignment.chunk_token_id2,
                alignment.chunk_text1,
                alignment.chunk_text2,
                align_types,
                alignment.align_score
            )
            return alignment_new

        pk3 = PromiseKeeperOnFuture(solve_core)

        def future_map3(t: MyFuture[Tuple[AlignmentLabelUnit, NLIPred, NLIPred]]) -> MyFuture[AlignmentLabelUnit]:
            return pk3.get_future(t)

        def future_map4(t: List[MyFuture[AlignmentLabelUnit]]) -> MyFuture[List[AlignmentLabelUnit]]:
            pass

        step1: List[AlignmentLabelUnit] = alignment_list
        step2: List[MyFuture[AlignmentLabelUnit, NLIPred, NLIPred]] = lmap(future_map1, step1)
        step3: List[MyFuture[Tuple[AlignmentLabelUnit, NLIPred, NLIPred]]] = lmap(future_map2, step2)
        step4: List[MyFuture[AlignmentLabelUnit]] = lmap(future_map3, step3)
        step5: MyFuture[List[AlignmentLabelUnit]] = future_map4(step4)
        return problem_id, step5

    def solve_one(self, problem: iSTSProblemWChunk) -> AlignmentPrediction:
        base_alignment: AlignmentPrediction = self.base_solver.solve_one(problem)
        problem_id, alignment_list = base_alignment
        NLIPred = List[float]

        pk = PromiseKeeper(self.predict_fn)
        my_future_list = []
        for alignment in alignment_list:
            if alignment.align_types[0] == ALIGN_NOALI:
                def get_dummy_future():
                    t = MyFuture()
                    t.set_value(None)
                    return t
                nli_p1_MyFuture: MyFuture[NLIPred] = get_dummy_future()
                nli_p2_MyFuture: MyFuture[NLIPred] = get_dummy_future()
            else:
                nli_p1_MyFuture: MyFuture[NLIPred] = pk.get_future((alignment.chunk_text1, alignment.chunk_text2))
                nli_p2_MyFuture: MyFuture[NLIPred] = pk.get_future((alignment.chunk_text2, alignment.chunk_text1))
            my_future_list.append((alignment, nli_p1_MyFuture, nli_p2_MyFuture))

        pk.do_duty()
        def is_entailment(probs):
            return probs[0] > 0.5

        def solve_core(alignment: AlignmentLabelUnit, nli_pred1: NLIPred, nli_pred2: NLIPred) -> AlignmentLabelUnit:
            if nli_pred1 is None or nli_pred2 is None:
                align_types = [ALIGN_NOALI]
            elif is_entailment(nli_pred1) and is_entailment(nli_pred2):
                align_types = [ALIGN_EQUI]
            elif is_entailment(nli_pred1) and not is_entailment(nli_pred2):
                align_types = [ALIGN_SPE1]
            elif not is_entailment(nli_pred1) and is_entailment(nli_pred2):
                align_types = [ALIGN_SPE2]
            else:
                align_types = [ALIGN_SIMI]

            alignment_new = AlignmentLabelUnit(
                alignment.chunk_token_id1,
                alignment.chunk_token_id2,
                alignment.chunk_text1,
                alignment.chunk_text2,
                align_types,
                alignment.align_score
            )
            return alignment_new

        out_alignment_list = []
        for alignment, f1, f2 in my_future_list:
            alignment_new = solve_core(alignment, f1.get(), f2.get())
            out_alignment_list.append(alignment_new)
        return problem_id, out_alignment_list
