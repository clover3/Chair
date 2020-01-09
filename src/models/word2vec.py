import os

import numpy

from cpath import data_path


def load_w2v_gensim(path):
    import gensim
    return gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)


def build_index(voca, w2v, k):
    print("building index..")
    n_reserve = 5
    voca_size = len(voca) + n_reserve
    idx2vect = numpy.zeros(shape=(voca_size, k), dtype='float32')

    # 0 is only predefined vector
    idx2vect[0] = numpy.random.uniform(-0.25,0.25,k)

    f = open("missing_w2v.txt", "w", errors='ignore')
    match_count = 0


    idx = n_reserve
    word2idx = {}
    for word in voca:
        word2idx[word] = idx
        if word in w2v:
            idx2vect[idx] = w2v[word]
            match_count = match_count +1
        else:
            f.write(word+"\n")
            idx2vect[idx] = numpy.zeros(k, dtype='float32')
        idx += 1
    f.close()
    print("w2v {} of {} matched".format(match_count, len(voca)))
    return idx2vect, word2idx


def load_w2v():
    path = os.path.join(data_path, "embeddings", "GoogleNews-vectors-negative300.bin")
    return load_w2v_gensim(path)



if __name__ == '__main__':
    load_w2v()