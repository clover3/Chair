
from typing import Iterable, Callable, Dict, Iterator
from typing import List, Tuple

from alignment.ists_eval.chunked_eval import ISTSChunkedSolver
from alignment.ists_eval.chunked_solver.exact_match_solver import score_chunk_pair_exact_match
from alignment.ists_eval.chunked_solver.solver_common import get_similarity_table
from alignment.ists_eval.chunked_solver.word2vec_solver import Word2VecChunkHelper
from alignment.ists_eval.prediction_helper import get_alignment_label_units
from dataset_specific.ists.parse import iSTSProblemWChunk, AlignmentPrediction, ALIGN_EQUI, ALIGN_SPE1, ALIGN_SPE2, \
    ALIGN_NOALI
from misc_lib import timed_lmap
from trainer.promise import MyFuture, PromiseKeeper
from trainer_v2.keras_server.name_short_cuts import NLIPredictorSig

NLIPred = List[float]
def get_sorted_table_scores(matrix) -> List[Tuple[int, int, float]]:
    items = []
    n_left = len(matrix)
    n_right = len(matrix[0])
    for i in range(n_left):
        for j in range(n_right):
            item = i, j, matrix[i][j]
            items.append(item)

    items.sort(key=lambda x: x[2], reverse=True)
    return items

def enum_chunk_pairs(alignment_list) -> Iterator[Tuple[str, str]]:
    for alignment in alignment_list:
        if alignment.align_types[0] != ALIGN_NOALI:
            yield alignment.chunk_text1, alignment.chunk_text2
            yield alignment.chunk_text2, alignment.chunk_text1


def greedy_alignment_building(problem: iSTSProblemWChunk,
                              top_k_pair_scores: List[Tuple[int, int, float]],
                              get_best_ali_by_entail: Callable[[str, str], str]):
    aligned_1 = set()
    aligned_2 = set()
    left_items = []
    right_items = []
    labels = []
    for i, j, score in top_k_pair_scores:
        if i not in aligned_1 and j not in aligned_2 and score > 0:
            best_label = get_best_ali_by_entail(problem.chunks1[i], problem.chunks2[j])
            if best_label is not ALIGN_NOALI:
                left_items.append(i)
                right_items.append(j)
                aligned_1.add(i)
                aligned_2.add(j)
                labels.append([best_label])
    for i in range(len(problem.chunks1)):
        if i not in aligned_1:
            left_items.append(i)
            aligned_1.add(i)
            right_items.append(None)
            labels.append([ALIGN_NOALI])
    for i in range(len(problem.chunks2)):
        if i not in aligned_2:
            left_items.append(None)
            right_items.append(i)
            aligned_2.add(i)
            labels.append([ALIGN_NOALI])
    ret = get_alignment_label_units(left_items, right_items, labels, problem)
    return ret

# Option 1: Work on aligned outputs of Exact match or word2vec
class PartialNLIDrivenSolver(ISTSChunkedSolver):
    def __init__(self, predict_fn: NLIPredictorSig, word2vec_path, verbose=False):
        self.predict_fn = predict_fn
        self.chunk_helper = Word2VecChunkHelper(word2vec_path)
        self.verbose = verbose

    def print(self, msg):
        if self.verbose:
            print(msg)

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

    def chunk_pair_similarity(self, c1, c2):
        s1 = score_chunk_pair_exact_match(c1, c2)
        s2 = self.chunk_helper.score_chunk_pair(c1, c2)
        return s1 + s2 * 0.1

    def solve_one(self, problem: iSTSProblemWChunk) -> AlignmentPrediction:
        self.print(str(problem))
        table: List[List[float]] = get_similarity_table(problem, self.chunk_pair_similarity)
        pair_scores: List[Tuple[int, int, float]] = get_sorted_table_scores(table)
        k = int(len(table) * 1.5)
        top_k_pair_scores = pair_scores[:k]

        self.print("Top-k pairs:")
        for i, j, s in top_k_pair_scores:
            msg = f"{i} ({problem.chunks1[i]}) : {j} ({problem.chunks2[j]}) {s}"
            self.print(msg)
        candi_pairs: List[Tuple[str, str]] = [(problem.chunks1[i], problem.chunks2[j]) for i,j, _ in top_k_pair_scores]
        text1 = problem.text1
        text2 = problem.text2

        def exclude(text, chunk):
            return text.replace(chunk, "[MASK]")

        pair_to_check = []
        pair_to_check.extend(candi_pairs)
        pair_to_check.extend([(c2, c1) for c1, c2 in candi_pairs])
        pair_to_check.extend([(exclude(text1, c1), c2) for c1, c2 in candi_pairs])
        pair_to_check.extend([(exclude(text2, c2), c1) for c1, c2 in candi_pairs])
        pair_to_check.extend([(text2, c1) for c1 in problem.chunks1])
        pair_to_check.extend([(text1, c2) for c2 in problem.chunks2])
        entail_d: Dict[Tuple[str, str], List[float]] = self.pred_entail(pair_to_check)

        def entail(t1, t2) -> bool:
            return entail_d[t1, t2][0] > 0.5

        def get_best_ali_by_entail(c1, c2):
            text1_wo_c1 = exclude(text1, c1)
            text2_wo_c2 = exclude(text2, c2)

            entail_12 = entail(c1, c2)
            if not entail_12:
                entail_12 = entail(text1, c2) and not entail(text1_wo_c1, c2)

            entail_21 = entail(c2, c1)
            if not entail_21:
                entail_21 = entail(text2, c1) and not entail(text2_wo_c2, c1)

            self.print(str(c1, c2, entail(c1, c2), entail(text1, c2), entail(text1_wo_c1, c2)))
            self.print(str(c2, c1, entail(c2, c1), entail(text2, c1), entail(text2_wo_c2, c1)))
            if entail_12 and entail_21:
                align_label = ALIGN_EQUI
            elif entail_12 and not entail_21:
                align_label = ALIGN_SPE1
            elif not entail_12 and entail_21:
                align_label = ALIGN_SPE2
            else:
                align_label = ALIGN_NOALI

            self.print("({}, {}) -> {}".format(c1, c2, align_label))
            return align_label

        ret = greedy_alignment_building(problem, top_k_pair_scores, get_best_ali_by_entail)
        return ret
