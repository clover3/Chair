
import os
import pickle
import sys
from functools import partial

import tensorflow as tf

import cpath
from data_generator.tokenizer_wo_tf import EncoderUnitOld
from models.transformer import hyperparams
from models.transformer.transformer_weight import transformer_weight
from trainer.model_saver import load_model_w_scope
from trainer.np_modules import flatten_from_batches
from trainer.promise import PromiseKeeper, MyPromise, list_future
from trainer.tf_train_module import init_session
from trainer.ukp.predict_arg_exp_clueweb import get_topic_from_path, CluewebTokenReader, batch_iter_from_entry_iter


class AgreePredictor:
    def __init__(self, hparam, voca_size, start_model_path):
        print("AgreePredictor")
        tf.reset_default_graph()
        self.task = transformer_weight(hparam, voca_size, False)
        self.sess = init_session()
        self.sess.run(tf.global_variables_initializer())
        load_model_w_scope(self.sess, start_model_path, ['bert', 'cls_dense'])

    def batch2feed_dict(self, batch):
        x0 ,x1 ,x2, y  = batch
        feed_dict = {
            self.task.x_list[0]: x0,
            self.task.x_list[1]: x1,
            self.task.x_list[2]: x2,
            self.task.y: y,
        }
        return feed_dict

    def run(self, batch):
        sout, = self.sess.run([self.task.sout],
                                   feed_dict=self.batch2feed_dict(batch))
        return sout, batch[0]


def run(token_path, ranked_list_path, start_model_path, output_path):
    voca_size = 30522
    target_topic = get_topic_from_path(ranked_list_path)
    print(target_topic)
    hp = hyperparams.HPBert()
    vocab_filename = "bert_voca.txt"
    voca_path = os.path.join(cpath.data_path, vocab_filename)
    batch_size = 256

    encoder = EncoderUnitOld(hp.seq_max, voca_path)

    def encode(tokens):
        seg_a = encoder.encoder.ft.convert_tokens_to_ids(tokens)
        d = encoder.encode_inner(seg_a, [])
        return d["input_ids"], d["input_mask"], d["segment_ids"], 0

    token_reader = CluewebTokenReader(target_topic, ranked_list_path, token_path)

    def predict_list(batch_size, tokens_list):
        predictor = AgreePredictor(hp, voca_size, start_model_path)
        entry_itr = [encode(tokens) for tokens in tokens_list]
        print("len(tokens_list)", len(tokens_list))
        result = []
        for idx, batch in enumerate(batch_iter_from_entry_iter(batch_size, entry_itr)):
            result.append(predictor.run(batch))
            if idx % 100 == 0:
                print(idx)
        r = flatten_from_batches(result)
        print("len(r)", len(r))
        return r

    # iterate token reader and schedule task with promise keeper
    pk = PromiseKeeper(partial(predict_list, batch_size))
    result_list = []
    for idx, doc in enumerate(token_reader.iter_docs()):
        future_list = []
        for sent in doc:
            promise = MyPromise(sent, pk)
            future_list.append(promise.future())
        result_list.append(future_list)

        if idx == 1000:
            break

    # encode promise into the batches and run them
    pk.do_duty()

    r = []
    for future_list in result_list:
        r.append(list_future(future_list))

    pickle.dump(r, open(output_path, "wb"))


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])