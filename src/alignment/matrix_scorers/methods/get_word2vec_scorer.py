import gensim
import os
import numpy as np


from alignment.matrix_scorers.methods.vector_similarity_scorer import VectorSimilarityScorer
from data_generator.tokenizer_wo_tf import get_tokenizer, ids_to_text
from typing import List, Iterable, Callable, Dict, Tuple, Set


def get_word2vec_scorer(word2vec_path) -> VectorSimilarityScorer:
    w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    tokenizer = get_tokenizer()

    cache = dict()
    not_found = set()

    def get_vector(ids: List[int]):
        word = ids_to_text(ids, tokenizer)
        if word in not_found:
            raise KeyError

        if word in cache:
            value = cache[word]
        elif word in w2v:
            value = w2v.get_vector(word, True)
            cache[word] = value
        elif word.capitalize() in w2v:
            value = w2v.get_vector(word.capitalize(), True)
            cache[word] = value
        elif word.upper() in w2v:
            value = w2v.get_vector(word.upper(), True)
            cache[word] = value
        else:
            not_found.add(word)
            raise KeyError
        return value

    def get_similarity(v1, v2):
        return np.dot(v1, v2)

    return VectorSimilarityScorer(get_vector, get_similarity)


def get_word2vec_scorer_from_d() -> VectorSimilarityScorer:
    return get_word2vec_scorer(get_word2vec_path())


def get_word2vec_path():
    word2vec_path = os.path.join("D:\\data\\embeddings\\GoogleNews-vectors-negative300.bin")
    return word2vec_path



if __name__ == "__main__":
    main()