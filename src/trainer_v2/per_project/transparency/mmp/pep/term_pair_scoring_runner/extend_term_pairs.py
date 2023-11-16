import sys
from typing import List, Iterable, Callable, Dict, Tuple, Set
from omegaconf import OmegaConf

from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import dict_value_map
from misc_lib import group_by, get_first
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.misc_common import read_term_pair_table_w_score
from trainer_v2.per_project.transparency.mmp.pep.evidence_encoder.basic_embedding.nn_helper import \
    get_embedding_nn_index


def get_knn_factory():
    dir_name = "bert_encoding_max"
    embeddings, index, words = get_embedding_nn_index(dir_name)
    def knn(word, k) -> List[str]:
        idx = words.index(word)
        query_vector = embeddings[idx]
        query_vector = query_vector.reshape(1, -1)  # Reshape for FAISS compatibility

        D, I = index.search(query_vector, k)
        # Display the results
        # print("Top 10 similar vectors (inner product and index):")
        neighbor_terms: List[str] = []
        for i in range(k):
            idx = I[0][i]
            word = words[idx]
            neighbor_terms.append(word)
            # print(f"{idx}: {word}, Inner product: {D[0][i]}")

        return neighbor_terms
    return knn


def bert_normalize(tokenizer, term):
    sb_word = tokenizer.wordpiece_tokenizer.tokenize(term)
    out_s = sb_word[0]
    for t in sb_word[1:]:
        if t[:2] == "##":
            out_s += t[2:]
        else:
            print("sb_word not expected", sb_word)
            raise ValueError()

    if out_s != term:
        print(f"Term normalize {term} -> {out_s}")

    return out_s


def main():
    k_q = 10  # Number of terms to from kNN of query term
    k_d = 10  # Number of terms to from kNN of document term
    conf_path = sys.argv[1]

    # confs/experiment_confs/extend_term_pair.yaml
    conf = OmegaConf.load(conf_path)
    term_pair_scores_path = conf.term_pair_scores_path
    self_score_path = conf.self_score_path
    items = tsv_iter(self_score_path)
    self_score_d = {}
    for q_term, q_term_same, score in items:
        assert q_term == q_term_same
        self_score_d[q_term] = float(score)

    itr = read_term_pair_table_w_score(term_pair_scores_path)
    grouped: Dict[str, List[Tuple]] = group_by(itr, get_first)
    get_knn: Callable[[str, int], List[str]] = get_knn_factory()
    for q_term, entries in grouped.items():
        entries: List[Tuple[str, str, float]] = entries
        self_score: float = self_score_d[q_term]
        t_promising = min(1.1, self_score)  # Maybe replace with relative to q_term?
        # If q_term itself is not in the list add
        new_d_term_candidates = set()
        try:
            new_d_term_candidates.update(get_knn(q_term, k_q))
        except ValueError as e:
            print(e)

        n_promising = 0
        for _q_term, d_term, score in entries:
            if score > t_promising:
                try:
                    n_promising += 1
                    new_d_term_candidates.update(get_knn(d_term, k_d))
                except ValueError as e:
                    print(e)
        # Add extends to neighbor of top scoring terms
        print(q_term, n_promising, len(new_d_term_candidates))


if __name__ == "__main__":
    main()
