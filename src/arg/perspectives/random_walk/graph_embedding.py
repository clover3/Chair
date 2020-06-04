import os
from typing import List, Callable, Any, NewType

import gensim.models

from cpath import output_path

GraphEmbeddingTrainer = NewType('GraphEmbeddingTrainer', Callable[[List[List[str]]], Any] )

#word2vec_path = os.path.join(data_path, "GoogleNews-vectors-negative300.bin")

word2vec_path = os.path.join(output_path, "word2vec_clueweb12_13B")

def train_word2vec(corpus: List[List[str]]):
    model = gensim.models.Word2Vec.load(word2vec_path)
    model.train(sentences=corpus, epochs=100, total_examples=len(corpus))
    return model


