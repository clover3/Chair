from typing import Dict, Collection, TypeVar, NewType

A = TypeVar('A')
Edges = NewType('Edges', Dict[A, Dict[A, float]])


def run_random_walk(edges: Edges,
                    vertice: Collection[A],
                    max_repeat,
                    p_reset,
                    ):
    p_transition = 1 - p_reset

    def same(before, after):
        same_thres = 1e-4
        assert (len(before) == len(after))
        before.sort(key=lambda x: x[0])
        after.sort(key=lambda x: x[0])
        for b, a in zip(before, after):
            (word_b, p_b) = b
            (word_a, p_a) = a
            assert (word_a == word_b)
            if abs(p_b - p_a) > same_thres:
                return False
        return True

    init_p = 1 / len(vertice)
    p_vertice = [(vertex, init_p) for vertex in vertice]
    n_repeat = 0
    while n_repeat < max_repeat:
        p_vertice_next = dict((vertex, p_reset * init_p) for vertex in vertice)
        for vertex, p in p_vertice:
            for target, edge_p in edges[vertex].items():
                p_vertice_next[target] += edge_p * p * p_transition
        p_vertice_next = list(p_vertice_next.items())
        n_repeat += 1
        if same(p_vertice, p_vertice_next):
            break
        p_vertice = p_vertice_next
    print(n_repeat)
    return dict((vertex, p) for vertex, p in p_vertice)