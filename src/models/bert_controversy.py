from trainer.experiment import Experiment
from data_generator.controversy import Ams18
from trainer.tf_module import init_session
from models.transformer.transformer_binary import transformer_binary
from models.transformer.hyperparams import HPBert
import numpy as np
import tensorflow as tf
from trainer.np_modules import get_batches_ex
import path

class BertPredictor:
    def __init__(self, name= "WikiContrv"):
        self.vocab_size = 30522
        self.vocab_filename = "bert_voca.txt"
        self.hp = HPBert()
        self.sess = init_session()
        self.data_loader = Ams18.DataLoader(self.hp.seq_max, self.vocab_filename, self.vocab_size)

        self.sess.run(tf.global_variables_initializer())
        self.model = transformer_binary(self.hp, self.vocab_size, True)

        loader = tf.train.Saver()
        loader.restore(self.sess, path.get_model_full_path(name))


    def score(self, docs):
        inputs = self.data_loader.encode_docs(docs)

        def forward_run(inputs):
            batches = get_batches_ex(inputs, self.hp.batch_size, 3)
            logit_list = []
            for batch in batches:
                x0, x1, x2 = batch
                logits,  = self.sess.run([self.model.sout, ],
                                           feed_dict={
                                               self.model.x_list[0]: x0,
                                               self.model.x_list[1]: x1,
                                               self.model.x_list[2]: x2,
                                           })
                logit_list.append(logits)
            return np.concatenate(logit_list)

        output = forward_run(inputs)[:,1]
        return output