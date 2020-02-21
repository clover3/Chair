import os
import pickle
import sys
from functools import partial

import tensorflow as tf

import cpath
from adhoc.galago import load_galago_ranked_list
from data_generator.argmining import ukp
from data_generator.common import get_tokenizer
from data_generator.tokenizer_wo_tf import EncoderUnit
from models.transformer import hyperparams
from models.transformer.tranformer_nli import transformer_nli
from trainer.model_saver import load_model_w_scope
from trainer.np_modules import get_batches_ex, flatten_from_batches
from trainer.promise import PromiseKeeper, MyPromise, list_future
from trainer.tf_train_module import init_session


class UkpExPredictor:
    def __init__(self, hparam, voca_size, start_model_path):
        print("run_ukp_ex")
        tf.reset_default_graph()
        self.task = transformer_nli(hparam, voca_size, 5, True)
        self.sess = init_session()
        self.sess.run(tf.global_variables_initializer())
        load_model_w_scope(self.sess, start_model_path, ['bert', 'cls_dense', 'aux_conflict'])

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
        logits, conf_logit, c_soft = self.sess.run([self.task.logits, self.task.conf_logits, self.task.conf_softmax],
                                                   feed_dict=self.batch2feed_dict(batch))
        return logits, conf_logit, c_soft, batch[0]


class CluewebTokenReader:
    def __init__(self, topic, ranked_list_path, token_file_path):
        ranked_list_d = load_galago_ranked_list(ranked_list_path)
        self.ranked_list = ranked_list_d["unk-0"]

        self.tokenizer = get_tokenizer()
        self.topic = topic
        self.tokens = pickle.load(open(token_file_path, "rb"))
        self.doc_idx = 0

    def iter_docs(self):
        for doc_id, rank, score in self.ranked_list:
            doc = self.tokens[doc_id]
            yield doc
        print()

def batch_iter_from_entry_iter(batch_size, entry_iter):
    batch = []
    for entry in entry_iter:
        batch.append(entry)
        if len(batch) == batch_size:
            yield get_batches_ex(batch, batch_size, 4)[0]
            batch = []

    if batch:
        yield get_batches_ex(batch, batch_size, 4)[0]


def get_topic_from_path(path):
    for topic in ukp.all_topics:
        if topic in path:
            return topic
    raise Exception()

def run(token_path, ranked_list_path, start_model_path, output_path):
    voca_size = 30522
    target_topic = get_topic_from_path(ranked_list_path)
    print(target_topic)
    hp = hyperparams.HPBert()
    vocab_filename = "bert_voca.txt"
    voca_path = os.path.join(cpath.data_path, vocab_filename)
    batch_size = 256

    encoder = EncoderUnit(hp.seq_max, voca_path)
    seg_b = encoder.encoder.encode(target_topic + " is good")

    def encode(tokens):
        seg_a = encoder.encoder.ft.convert_tokens_to_ids(tokens)
        d = encoder.encode_inner(seg_a, seg_b)
        return d["input_ids"], d["input_mask"], d["segment_ids"], 0

    token_reader = CluewebTokenReader(target_topic, ranked_list_path, token_path)

    def predict_list(batch_size, tokens_list):
        predictor = UkpExPredictor(hp, voca_size, start_model_path)
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