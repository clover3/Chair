from typing import List

from dataset_specific.ists.parse import iSTSProblemWChunk, AlignmentPrediction, ALIGN_EQUI, ALIGN_SPE1, ALIGN_SPE2, \
    AlignmentLabelUnit, ALIGN_SIMI, ALIGN_NOALI
from trainer.promise import MyFuture, PromiseKeeper


# Steps
def code():
    # Init alignments as NOALI
    # Check if each chunk is entailed by the other sentence
    # While Break
    span_to_sent_entail_info = NotImplemented
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
                nli_p1_MyFuture: MyFuture[NLIPred] = pk.get_future(
                    (alignment.chunk_text1, alignment.chunk_text2))
                nli_p2_MyFuture: MyFuture[NLIPred] = pk.get_future(
                    (alignment.chunk_text2, alignment.chunk_text1))
            my_future_list.append((alignment, nli_p1_MyFuture, nli_p2_MyFuture))

        pk.do_duty()

        def is_entailment(probs):
            return probs[0] > 0.5

        def solve_core(alignment: AlignmentLabelUnit, nli_pred1: NLIPred,
                       nli_pred2: NLIPred) -> AlignmentLabelUnit:
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
