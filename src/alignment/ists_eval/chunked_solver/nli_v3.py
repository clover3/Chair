from abc import abstractmethod, ABC
from typing import Iterable, Callable, Dict, Iterator
from typing import List, Tuple

from alignment.ists_eval.chunked_eval import ISTSChunkedSolver
from alignment.ists_eval.chunked_solver.nli_partial import NLIPred, get_sorted_table_scores, greedy_alignment_building
from alignment.ists_eval.chunked_solver.solver_common import get_similarity_table
from alignment.ists_eval.chunked_solver.word2vec_solver import Word2VecChunkHelper
from cpath import word2vec_path
from dataset_specific.ists.parse import iSTSProblemWChunk, AlignmentPrediction, ALIGN_EQUI, ALIGN_SPE1, ALIGN_SPE2, \
    ALIGN_NOALI
from misc_lib import timed_lmap
from trainer.promise import MyFuture, PromiseKeeper
from trainer_v2.chair_logging import c_log
from trainer_v2.keras_server.name_short_cuts import NLIPredictorSig, get_nli14_cache_client, get_pep_cache_client


def exclude(text, chunk):
    return text.replace(chunk, "[MASK]")


class ISTSLabelPredictorI(ABC):
    @abstractmethod
    def generate_check_pairs(self, problem: iSTSProblemWChunk, candidate_pairs):
        pass

    @abstractmethod
    def classify(self, problem: iSTSProblemWChunk, c1, c2, entail: Callable):
        pass


class ISTSLabelPredictorWContext(ISTSLabelPredictorI):
    def generate_check_pairs(self, problem: iSTSProblemWChunk, candi_pairs: List[Tuple[str, str]]):
        text1 = problem.text1
        text2 = problem.text2
        pair_to_check: List[Tuple[str, str]] = []
        pair_to_check.extend(candi_pairs)
        pair_to_check.extend([(c2, c1) for c1, c2 in candi_pairs])
        pair_to_check.extend([(exclude(text1, c1), c2) for c1, c2 in candi_pairs])
        pair_to_check.extend([(exclude(text2, c2), c1) for c1, c2 in candi_pairs])
        pair_to_check.extend([(text2, c1) for c1 in problem.chunks1])
        pair_to_check.extend([(text1, c2) for c2 in problem.chunks2])
        return pair_to_check

    def classify(self, problem: iSTSProblemWChunk, c1, c2, entail: Callable):
        text1 = problem.text1
        text2 = problem.text2

        text1_wo_c1 = exclude(text1, c1)
        text2_wo_c2 = exclude(text2, c2)

        entail_12 = entail(c1, c2)
        if not entail_12:
            entail_12 = entail(text1, c2) and not entail(text1_wo_c1, c2)

        entail_21 = entail(c2, c1)
        if not entail_21:
            entail_21 = entail(text2, c1) and not entail(text2_wo_c2, c1)

        if entail_12 and entail_21:
            align_label = ALIGN_EQUI
        elif entail_12 and not entail_21:
            align_label = ALIGN_SPE1
        elif not entail_12 and entail_21:
            align_label = ALIGN_SPE2
        else:
            align_label = ALIGN_NOALI

        c_log.debug("({}, {}) -> {}".format(c1, c2, align_label))
        return align_label


class ISTSLabelPredictorWOContext(ISTSLabelPredictorI):
    def generate_check_pairs(self, problem: iSTSProblemWChunk, candi_pairs: List[Tuple[str, str]]):
        pair_to_check = []
        pair_to_check.extend(candi_pairs)
        pair_to_check.extend([(c2, c1) for c1, c2 in candi_pairs])
        return pair_to_check

    def classify(self, problem: iSTSProblemWChunk, c1, c2, entail: Callable) -> str:
        entail_12 = entail(c1, c2)
        entail_21 = entail(c2, c1)
        if entail_12 and entail_21:
            align_label = ALIGN_EQUI
        elif entail_12 and not entail_21:
            align_label = ALIGN_SPE1
        elif not entail_12 and entail_21:
            align_label = ALIGN_SPE2
        else:
            align_label = ALIGN_NOALI

        c_log.debug("({}, {}) -> {}".format(c1, c2, align_label))
        return align_label


# Option 1: Work on aligned outputs of Exact match or word2vec
class NLISolver3(ISTSChunkedSolver):
    def __init__(self,
                 predict_fn: NLIPredictorSig,
                 get_similar_fn: Callable,
                 ists_label_predictor: ISTSLabelPredictorI,
                 ):
        self.predict_fn = predict_fn
        self.get_similar_fn = get_similar_fn
        self.label_predictor = ists_label_predictor

    def batch_solve(self, problems: List[iSTSProblemWChunk]) -> List[AlignmentPrediction]:
        return timed_lmap(self.solve_one, problems)

    def enum_chunk_to_text(self, problem: iSTSProblemWChunk) -> Iterator[Tuple[str, str]]:
        for c1 in problem.chunks1:
            yield problem.text2, c1
        for c2 in problem.chunks2:
            yield problem.text1, c2

    def pred_entail(self, text_pair: Iterable[Tuple[str, str]]) -> Dict[Tuple[str, str], List[float]]:
        pk = PromiseKeeper(self.predict_fn)
        my_future_list = []
        for s1, s2 in text_pair:
            f1: MyFuture[NLIPred] = pk.get_future((s1, s2))
            my_future_list.append(((s1, s2), f1))
        pk.do_duty()
        entail_d: Dict[Tuple[str, str], List[float]] = {}
        for (s1, s2), f1 in my_future_list:
            entail_d[s1, s2] = f1.get()
        return entail_d

    def solve_one(self, problem: iSTSProblemWChunk) -> AlignmentPrediction:
        c_log.debug(str(problem))
        table: List[List[float]] = get_similarity_table(problem, self.get_similar_fn)
        pair_scores: List[Tuple[int, int, float]] = get_sorted_table_scores(table)
        k = int(len(table) * 1.5)
        top_k_pair_scores = pair_scores[:k]

        c_log.debug("Top-k pairs:")
        for i, j, s in top_k_pair_scores:
            msg = f"{i} ({problem.chunks1[i]}) : {j} ({problem.chunks2[j]}) {s}"
            c_log.debug(msg)
        candi_pairs: List[Tuple[str, str]] = [(problem.chunks1[i], problem.chunks2[j]) for i, j, _ in top_k_pair_scores]
        pair_to_check = self.label_predictor.generate_check_pairs(problem, candi_pairs)
        entail_d: Dict[Tuple[str, str], List[float]] = self.pred_entail(pair_to_check)

        def entail(t1, t2) -> bool:
            return entail_d[t1, t2][0] > 0.5

        def classify(c1, c2):
            return self.label_predictor.classify(problem, c1, c2, entail)

        ret = greedy_alignment_building(problem, top_k_pair_scores, classify)
        return ret


# Option 1: Work on aligned outputs of Exact match or word2vec
class NLISolver3Batch(ISTSChunkedSolver):
    def __init__(self,
                 predict_fn: NLIPredictorSig,
                 get_similar_fn: Callable,
                 ists_label_predictor: ISTSLabelPredictorI,
                 ):
        self.predict_fn = predict_fn
        self.get_similar_fn = get_similar_fn
        self.label_predictor = ists_label_predictor

    def batch_solve(self, problems: List[iSTSProblemWChunk]) -> List[AlignmentPrediction]:
        pairs_to_check = []
        for p in problems:
            pairs_to_check.extend(self.generate_check_pairs(p))

        entail_d: Dict[Tuple[str, str], List[float]] = self.pred_entail(pairs_to_check)

        def solve(p: iSTSProblemWChunk):
            return self.solve_one(p, entail_d)
        return timed_lmap(solve, problems)

    def enum_chunk_to_text(self, problem: iSTSProblemWChunk) -> Iterator[Tuple[str, str]]:
        for c1 in problem.chunks1:
            yield problem.text2, c1
        for c2 in problem.chunks2:
            yield problem.text1, c2

    def pred_entail(self, text_pair: Iterable[Tuple[str, str]]) -> Dict[Tuple[str, str], List[float]]:
        pk = PromiseKeeper(self.predict_fn)
        my_future_list = []
        for s1, s2 in text_pair:
            f1: MyFuture[NLIPred] = pk.get_future((s1, s2))
            my_future_list.append(((s1, s2), f1))
        pk.do_duty(True)
        entail_d: Dict[Tuple[str, str], List[float]] = {}
        for (s1, s2), f1 in my_future_list:
            entail_d[s1, s2] = f1.get()
        return entail_d

    def generate_check_pairs(self, problem: iSTSProblemWChunk) -> List[Tuple[str, str]]:
        c_log.debug(str(problem))
        table: List[List[float]] = get_similarity_table(problem, self.get_similar_fn)
        pair_scores: List[Tuple[int, int, float]] = get_sorted_table_scores(table)
        k = int(len(table) * 1.5)
        top_k_pair_scores = pair_scores[:k]

        c_log.debug("Top-k pairs:")
        for i, j, s in top_k_pair_scores:
            msg = f"{i} ({problem.chunks1[i]}) : {j} ({problem.chunks2[j]}) {s}"
            c_log.debug(msg)
        candi_pairs: List[Tuple[str, str]] = [(problem.chunks1[i], problem.chunks2[j]) for i, j, _ in top_k_pair_scores]
        pair_to_check: List[Tuple[str, str]] = self.label_predictor.generate_check_pairs(problem, candi_pairs)
        return pair_to_check

    def solve_one(self, problem: iSTSProblemWChunk, entail_d):
        table: List[List[float]] = get_similarity_table(problem, self.get_similar_fn)
        pair_scores: List[Tuple[int, int, float]] = get_sorted_table_scores(table)
        k = int(len(table) * 1.5)
        top_k_pair_scores = pair_scores[:k]

        def entail(t1, t2) -> bool:
            return entail_d[t1, t2][0] > 0.5

        def classify(c1, c2):
            return self.label_predictor.classify(problem, c1, c2, entail)

        ret = greedy_alignment_building(problem, top_k_pair_scores, classify)
        return ret


def get_solver(nli_model_type, label_classifier_type) -> NLISolver3Batch:
    chunk_helper = Word2VecChunkHelper(word2vec_path)
    nli_predictor_d = {
        'base': get_nli14_cache_client,
        'pep': get_pep_cache_client
    }

    label_predictor_d = {
        'w_context': ISTSLabelPredictorWContext,
        'wo_context': ISTSLabelPredictorWOContext
    }

    nli_predictor = nli_predictor_d[nli_model_type]()
    ists_label_predictor = label_predictor_d[label_classifier_type]()

    return NLISolver3Batch(
        nli_predictor,
        chunk_helper.chunk_pair_similarity_w_em,
        ists_label_predictor
    )
