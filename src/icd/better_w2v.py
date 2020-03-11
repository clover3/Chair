from gensim.models.word2vec import train_cbow_pair, Word2Vec, train_sg_pair
from numpy import sum as np_sum


def train_batch(model, sentences, alpha, work=None, neu1=None, compute_loss=False):
    """Update CBOW model by training on a sequence of sentences.

    Called internally from :meth:`~gensim.models.word2vec.Word2Vec.train`.

    Warnings
    --------
    This is the non-optimized, pure Python version. If you have a C compiler, Gensim
    will use an optimized code path from :mod:`gensim.models.word2vec_inner` instead.

    Parameters
    ----------
    model : :class:`~gensim.models.word2vec.Word2Vec`
        The Word2Vec model instance to train.
    sentences : iterable of list of str
        The corpus used to train the model.
    alpha : float
        The learning rate
    work : object, optional
        Unused.
    neu1 : object, optional
        Unused.
    compute_loss : bool, optional
        Whether or not the training loss should be computed in this batch.

    Returns
    -------
    int
        Number of words in the vocabulary actually used for training (that already existed in the vocabulary
        and were not discarded by negative sampling).

    """
    result = 0
    for sentence in sentences:
        word_vocabs = [
            model.wv.vocab[w] for w in sentence if w in model.wv.vocab
                                                   and model.wv.vocab[w].sample_int > model.random.rand() * 2 ** 32
        ]
        word = word_vocabs[0]
        start = 1
        window_pos = enumerate(word_vocabs[start:], start)
        word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None)]
        l1 = np_sum(model.wv.syn0[word2_indices], axis=0)  # 1 x vector_size
        if word2_indices and model.cbow_mean:
            l1 /= len(word2_indices)
        train_cbow_pair(model, word, word2_indices, l1, alpha, compute_loss=compute_loss)
        for word2idx in word2_indices:
            train_sg_pair(
                model, model.wv.index2word[word.index], word2idx, alpha, compute_loss=compute_loss
            )
            train_sg_pair(
                model, model.wv.index2word[word2idx], word.index, alpha, compute_loss=compute_loss
            )

        result += len(word_vocabs)
    return result


class Code2Vec(Word2Vec):
    def __init__(self, sentences: object) -> object:
        print("Code2Vec __init")
        super(Code2Vec, self).__init__(sentences=sentences, window=999, min_count=1, negative=5,
                                       )

    def _do_train_job(self, sentences, alpha, inits):
        """Train the model on a single batch of sentences.

        Parameters
        ----------
        sentences : iterable of list of str
            Corpus chunk to be used in this training batch.
        alpha : float
            The learning rate used in this batch.
        inits : (np.ndarray, np.ndarray)
            Each worker threads private work memory.

        Returns
        -------
        (int, int)
             2-tuple (effective word count after ignoring unknown words and sentence length trimming, total word count).

        """
        work, neu1 = inits
        tally = 0
        tally += train_batch(self, sentences, alpha, work, self.compute_loss)
        return tally, self._raw_word_count(sentences)

