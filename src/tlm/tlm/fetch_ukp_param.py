import os
import pickle

import tensorflow as tf

from data_generator.argmining.ukp import BertDataLoader
from models.transformer import hyperparams
from path import get_model_full_path, output_path, get_bert_full_path
from tlm.param_analysis import fetch_hidden_vector


def do_fetch_value():
    hp = hyperparams.HPBert()
    voca_size = 30522
    encode_opt = "is_good"
    topic = "abortion"
    tt_run_name = "arg_nli_{}_is_good".format(topic)
    run_name = "arg_{}_{}_{}".format("fetch_value", topic, encode_opt)
    data_loader = BertDataLoader(topic, True, hp.seq_max, "bert_voca.txt", option=encode_opt)

    model_path = get_model_full_path(tt_run_name)
    r = fetch_hidden_vector(hp, voca_size, run_name, data_loader, model_path)
    pickle.dump(r, open(os.path.join(output_path, "hv_tt.pickle"), "wb"))

    tf.reset_default_graph()
    model_path = get_bert_full_path()
    r = fetch_hidden_vector(hp, voca_size, run_name, data_loader, model_path)
    pickle.dump(r, open(os.path.join(output_path, "hv_lm.pickle"), "wb"))


do_fetch_value()