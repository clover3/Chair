import random

import faiss
import numpy as np
from cache import load_pickle_from
from cpath import output_path
from list_lib import left, right
from misc_lib import path_join, TimeEstimator
from trainer_v2.chair_logging import c_log
from typing import List, Iterable, Callable, Dict, Tuple, Set


def load_emb(dir_name, st, ed):
    paired: List[Tuple[str, np.ndarray]] = []
    for job_no in range(st, ed):
        pickle_path = path_join(output_path, "mmp", dir_name, str(job_no))
        words, embeddings = load_pickle_from(pickle_path)
        for w, e in zip(words, embeddings):
            paired.append((w, e))
    return paired


def main():
    dir_name = "bert_encoding_max"
    c_log.info("Loading vectors from %s", dir_name)
    word_emb: List[Tuple[str, np.ndarray]] = load_emb(dir_name, 0, 10)
    c_log.info("Done")
    save_path = path_join(output_path, "mmp", dir_name + "_sim_arr.txt")

    words: List[str] = left(word_emb)
    embeddings: List[np.ndarray] = right(word_emb)

    vector_dim, = embeddings[0].shape

    # Create a FAISS index - Here, I'm using the IndexFlatIP which is for inner product
    index = faiss.IndexFlatIP(vector_dim)

    embeddings = np.array(embeddings)
    faiss.normalize_L2(embeddings)

    # Adding the database vectors to the index
    index.add(embeddings)
    random.seed(0)
    f_out = open(save_path, "w")
    ticker = TimeEstimator(100000)
    for idx, target_word in enumerate(words):
        ticker.tick()
        query_vector = embeddings[idx]
        query_vector = query_vector.reshape(1, -1)  # Reshape for FAISS compatibility

        # Number of top similar vectors to retrieve
        k = 1000
        # Performing the search
        # D, I = index.search(n=n, x=query_vector, k=k, distances=dist, labels=labels)  # D is the distance, I is the index of vectors
        D, I = index.search(query_vector, k)
        # Display the results
        # print("Top 10 similar vectors (inner product and index):")
        neighbor_terms = []
        for i in range(k):
            idx = I[0][i]
            word = words[idx]
            neighbor_terms.append(word)
            # print(f"{idx}: {word}, Inner product: {D[0][i]}")

        # print("{}: {} ".format(target_word, neighbor_terms))
        # print(target_word)
        out_row = [target_word] + neighbor_terms
        # print("\t".join(neighbor_terms))
        f_out.write("\t".join(out_row))


if __name__ == "__main__":
    main()