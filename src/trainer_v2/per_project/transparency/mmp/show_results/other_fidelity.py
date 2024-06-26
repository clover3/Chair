import time
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator


def top_k_overlap(scores1: List[float], scores2: List[float], k=10) -> float:
    top_k_indices1 = sorted(range(len(scores1)), key=lambda i: scores1[i], reverse=True)[:k]
    top_k_indices2 = sorted(range(len(scores2)), key=lambda i: scores2[i], reverse=True)[:k]

    k_effective = min(k, len(scores1))
    overlap = len(set(top_k_indices1) & set(top_k_indices2))
    overlap_ratio = overlap / k_effective
    return overlap_ratio


from typing import List

def pairwise_preference_match(scores1: List[float], scores2: List[float]) -> float:
    if len(scores1) == 1 and len(scores2) == 1:
        return 1.0

    if len(scores1) != len(scores2):
        raise ValueError("scores1 and scores2 must have the same length")

    # Calculate the number of maintained preference pairs
    maintained_pairs = 0
    total_pairs = (len(scores1) * (len(scores1) - 1)) // 2

    for i in range(len(scores1)):
        for j in range(i + 1, len(scores1)):
            if (scores1[i] >= scores1[j] and scores2[i] >= scores2[j]) or \
               (scores1[i] <= scores1[j] and scores2[i] <= scores2[j]):
                maintained_pairs += 1

    # Calculate the fraction of maintained preference pairs
    fraction_maintained = maintained_pairs / total_pairs

    return fraction_maintained



# Test cases
def test_pairwise_preference_match():
    scores1 = [0.9, 0.8, 0.7, 0.6, 0.5]
    scores2 = [0.95, 0.85, 0.75, 0.65, 0.55]
    assert pairwise_preference_match(scores1, scores2) == 1.0

    scores1 = [0.9, 0.8, 0.7, 0.6, 0.5]
    scores2 = [0.5, 0.6, 0.7, 0.8, 0.9]
    assert pairwise_preference_match(scores1, scores2) == 0.0

    scores1 = [0.9, 0.8, 0.7, 0.6, 0.5]
    scores2 = [0.5, 0.4, 0.3, 0.2, 0.1]
    assert pairwise_preference_match(scores1, scores2) == 1.0

    scores1 = [0.9, 0.8, 0.7, 0.6, 0.5]
    scores2 = [0.95, 0.75, 0.85, 0.65, 0.55]
    assert pairwise_preference_match(scores1, scores2) == 0.8

    scores1 = [0.9, 0.8, 0.7, 0.6, 0.5]
    scores2 = [0.95, 0.55, 0.75, 0.65, 0.85]
    assert pairwise_preference_match(scores1, scores2) == 0.6

    scores1 = [0.9]
    scores2 = [0.8]
    assert pairwise_preference_match(scores1, scores2) == 1.0

    scores1 = [0.9, 0.8, 0.7]
    scores2 = [0.8, 0.9]
    try:
        pairwise_preference_match(scores1, scores2)
        assert False, "Expected ValueError for lists with different lengths"
    except ValueError:
        pass

    print("All tests passed!")

# Run the tests
def test_top_k_overlap():
    scores1 = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    scores2 = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]
    assert top_k_overlap(scores1, scores2, k=5) == 1.0

    scores1 = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    scores2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    assert top_k_overlap(scores1, scores2, k=5) == 0.0

    scores1 = [0.9, 0.8, 0.7, 0.6, 0.5]
    scores2 = [0.5, 0.6, 0.7, 0.8, 0.9]
    assert top_k_overlap(scores1, scores2, k=3) == 1 / 3

    scores1 = [0.9, 0.8, 0.7, 0.6, 0.5]
    scores2 = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert top_k_overlap(scores1, scores2, k=3) == 1/ 3

    scores1 = [0.9]
    scores2 = [0.9]
    assert top_k_overlap(scores1, scores2) == 1 / 1

    print("All tests passed!")


def test_pairwise_preference_time():
    scores1 = [0.9 for _ in range(1000)]
    scores2 = [0.9 for _ in range(1000)]

    st = time.time()
    pairwise_preference_match(scores1, scores2)
    ed = time.time()
    time_per_q = ed-st
    all_q_time = time_per_q * 1000
    all_method_time = all_q_time * 10
    print("{}".format(ed-st))
    print("{}sec={}hours".format(all_method_time, all_method_time / 3600))


def main():
    test_pairwise_preference_time()


if __name__ == "__main__":
    main()