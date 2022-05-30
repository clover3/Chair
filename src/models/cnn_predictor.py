import numpy as np
import tensorflow as tf

import cache
import cpath
from data_generator.old_codes.cnn import *
from models.cnn import CNN
from models.transformer import hyperparams
from summarization.tokenizer import *
from trainer.np_modules import get_batches_ex
from trainer.tf_module import init_session


class CNNPredictor:
    def __init__(self, name= "WikiContrvCNN", input_name=None):
        if input_name is None:
            input_name = name
        self.hp = hyperparams.HPCNN()
        self.sess = init_session()
        self.sess.run(tf.global_variables_initializer())
        self.dropout_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.seq_max = self.hp.seq_max
        self.word2idx = cache.load_cache(input_name+".voca")
        init_emb = cache.load_cache("init_emb_word2vec")
        self.model = CNN("controv", self.seq_max, 2, [2, 3, 4], 128,
                         init_emb, self.hp.embedding_size, self.dropout_prob)
        self.input_text = tf.placeholder(tf.int32,
                                       shape=[None, self.seq_max],
                                       name="comment_input")
        self.sout = self.model.network(self.input_text)
        self.tokenize = lambda x: tokenize(x, set(), False)

        loader = tf.train.Saver()
        loader.restore(self.sess, cpath.get_model_full_path(name))

    def encode(self, docs):

        def convert(word):
            if word in self.word2idx:
                return self.word2idx[word]
            else:
                return OOV

        data = []
        for doc in docs:
            entry = []
            for token in self.tokenize(doc):
                entry.append(convert(token))
            entry = entry[:self.seq_max]
            while len(entry) < self.seq_max:
                entry.append(PADDING)
            data.append((entry, 0))
        return data

    def score(self, docs):
        inputs = self.encode(docs)

        def forward_run(inputs):
            batches = get_batches_ex(inputs, self.hp.batch_size, 2)
            logit_list = []
            for batch in batches:
                x, y, = batch
                logits,  = self.sess.run([self.sout, ],
                                           feed_dict={
                                               self.input_text: x,
                                               self.dropout_prob: 1.0,
                                           })
                logit_list.append(logits)
            return np.concatenate(logit_list)

        output = forward_run(inputs)[:,1]
        return output