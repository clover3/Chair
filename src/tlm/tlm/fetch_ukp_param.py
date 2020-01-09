import os
import pickle

from cpath import get_model_full_path, output_path
from data_generator.argmining.ukp import BertDataLoader
from models.transformer import hyperparams
from tlm.param_analysis import fetch_params


def do_fetch_param():
    hp = hyperparams.HPBert()
    voca_size = 30522
    encode_opt = "is_good"
    topic = "abortion"
    load_run_name = "arg_nli_{}_is_good".format(topic)
    run_name = "arg_{}_{}_{}".format("fetch_grad", topic, encode_opt)
    data_loader = BertDataLoader(topic, True, hp.seq_max, "bert_voca.txt", option=encode_opt)
    model_path = get_model_full_path(load_run_name)
    names, vars = fetch_params(hp, voca_size, run_name, data_loader, model_path)
    r = names, vars
    pickle.dump(r, open(os.path.join(output_path, "params.pickle"), "wb"))




do_fetch_value()