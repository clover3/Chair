from typing import Callable, List

from dataset_specific.ists.parse import iSTSProblemWChunk


def get_similarity_table(problem: iSTSProblemWChunk,
                         score_chunk_pair: Callable[[str, str], float]) -> List[List[float]]:
    n_chunk1 = len(problem.chunks1)
    n_chunk2 = len(problem.chunks2)
    table = []
    for i in range(n_chunk1):
        arr = []
        for j in range(n_chunk2):
            chunk1: str = problem.chunks1[i]
            chunk2: str = problem.chunks2[j]
            arr.append(score_chunk_pair(chunk1, chunk2))
        table.append(arr)
    return table
