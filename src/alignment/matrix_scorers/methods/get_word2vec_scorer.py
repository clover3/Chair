import gensim
import os
import numpy as np


from alignment.matrix_scorers.methods.vector_similarity_scorer import VectorSimilarityScorer
from data_generator.tokenizer_wo_tf import get_tokenizer, ids_to_text
from typing import List, Iterable, Callable, Dict, Tuple, Set


def get_word2vec_scorer(word2vec_path) -> VectorSimilarityScorer:
    w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    tokenizer = get_tokenizer()

    def get_vector(ids: List[int]):
        word = ids_to_text(ids, tokenizer)
        return w2v.get_vector(word, True)

    def get_similarity(v1, v2):
        return np.dot(v1, v2)

    return VectorSimilarityScorer(get_vector, get_similarity)


def get_word2vec_path():
    word2vec_path = os.path.join("D:\\data\\embeddings\\GoogleNews-vectors-negative300.bin")
    return word2vec_path



if __name__ == "__main__":
    main()