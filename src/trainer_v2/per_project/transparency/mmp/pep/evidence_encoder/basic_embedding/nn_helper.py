from typing import List, Tuple

import faiss
import numpy as np

from cache import load_pickle_from
from cpath import output_path
from list_lib import left, right
from misc_lib import path_join
from trainer_v2.chair_logging import c_log


def load_emb(dir_name, st, ed):
    paired: List[Tuple[str, np.ndarray]] = []
    for job_no in range(st, ed):
        pickle_path = path_join(output_path, "mmp", dir_name, str(job_no))
        words, embeddings = load_pickle_from(pickle_path)
        for w, e in zip(words, embeddings):
            paired.append((w, e))
    return paired


def get_embedding_nn_index(dir_name):
    c_log.info("Loading vectors from %s", dir_name)
    word_emb: List[Tuple[str, np.ndarray]] = load_emb(dir_name, 0, 10)
    c_log.info("Done")
    words: List[str] = left(word_emb)
    embeddings: List[np.ndarray] = right(word_emb)
    vector_dim, = embeddings[0].shape
    # Create a FAISS index - Here, I'm using the IndexFlatIP which is for inner product
    index = faiss.IndexFlatIP(vector_dim)
    embeddings = np.array(embeddings)
    faiss.normalize_L2(embeddings)
    # Adding the database vectors to the index
    index.add(embeddings)
    return embeddings, index, words