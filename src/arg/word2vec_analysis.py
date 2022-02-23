import os

import gensim
import numpy as np

import tlm.qtype.partial_relevance.eval_metric_binary.eval_v3_common
import tlm.qtype.partial_relevance.eval_metric_binary.replace_v3
from cache import load_pickle_from
from cpath import output_path


def main():
    load_path = os.path.join(output_path, "word2vec_clueweb12_13B")

    model: gensim.models.Word2Vec = load_pickle_from(os.path.join(load_path))
    print(model.trainables.syn1neg.shape)
    terms = ['proposition', 'issue', 'reason']

    v_sum = np.sum([model[t] for t in terms], axis=0)
    print(v_sum)
    j = np.argmax(v_sum)
    print(list([model[t][j] for t in terms]))
    candi = model.wv.similar_by_vector(v_sum, topn=300)
    j_rank = np.argsort([model[word][j] for word, _ in candi])[::-1]
    for j_idx in j_rank[:20]:
        print(candi[j_idx])

    word = terms[0]
    term_id = model.wv.vocab[word].index
    #print(word, model.wv.vectors[term_id], model[word])

    scores = np.dot(v_sum, tlm.qtype.partial_relevance.eval_metric_binary.eval_v3_common.T)
    print(scores.shape)
    rank_by_context = np.argsort(scores)[::-1]
    for j_idx in rank_by_context[:20]:
        print(model.wv.index2word[j_idx])

if __name__ == "__main__":
    main()
