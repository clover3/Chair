from typing import List, Callable, Any, NewType

import gensim.models

GraphEmbeddingTrainer = NewType('GraphEmbeddingTrainer', Callable[[List[List[str]]], Any] )


def train_word2vec(corpus: List[List[str]]):
    model = gensim.models.Word2Vec(sentences=corpus,
                                   iter=200,
                                   min_count=5)
    return model


