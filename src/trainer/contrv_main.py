import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


from task.transformer_est import Transformer, Classification
from models.transformer import bert
from models.transformer import hyperparams
from data_generator.shared_setting import NLI
from trainer.tf_module import *
import tensorflow as tf
from data_generator.adhoc import ws
from data_generator.controversy import mscore, title, protest, Ams18
from data_generator.controversy import agree
from trainer.experiment import Experiment
from trainer.ExperimentConfig import ExperimentConfig
from trainer.controversy_experiments import ControversyExperiment
import evals.controversy


def train_mscore_regression():
    hp = hyperparams.HPMscore()

    e = Experiment(hp)

    e_config = ExperimentConfig()
    e_config.name = "Contrv_{}".format("C")
    e_config.num_epoch = 4
    e_config.save_interval = 10 * 60  # 10 minutes
    e_config.load_names = ['bert']
    vocab_size = 30522
    vocab_filename = "bert_voca.txt"

    data_loader = mscore.DataLoader(hp.seq_max, vocab_filename, vocab_size)
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    e.train_controversy_classification(e_config, data_loader, load_id)


def mscore_eval():
    hp = hyperparams.HPMscore()

    e = Experiment(hp)

    e_config = ExperimentConfig()
    e_config.name = "Contrv_{}".format("B_eval")
    e_config.num_epoch = 4
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert', 'reg_dense']
    vocab_size = 30522
    vocab_filename = "bert_voca.txt"

    data_loader = mscore.DataLoader(hp.seq_max, vocab_filename, vocab_size)
    load_id = ("Contrv_B", 'model-3001')
    #load_id = ("Contrv_B", 'model-6006')

    e.test_controversy_mscore(e_config, data_loader, load_id)




def contrv_pred():
    hp = hyperparams.HPQL()

    e = Experiment(hp)

    e_config = ExperimentConfig()
    e_config.name = "Contrv_{}".format("B")
    e_config.num_epoch = 4
    e_config.save_interval = 30 * 60  # 30 minuteslm_protest
    e_config.load_names = ['bert', 'cls']
    vocab_size = 30522
    vocab_filename = "bert_voca.txt"

    data_loader = ws.DataLoader(hp.seq_max, vocab_filename, vocab_size)
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    e.controv_lm(e_config, data_loader, load_id)


def cie():
    hp = hyperparams.HPCIE()
    e = Experiment(hp)
    e_config = ExperimentConfig()
    e_config.name = "cie"
    e_config.num_epoch = 40
    e_config.save_interval = 10 * 60  # 30 minutes
    e_config.load_names = ['bert']
    vocab_size = 30522
    vocab_filename = "bert_voca.txt"

    is_span = 1
    data_loader = title.DataLoader(hp.seq_max, vocab_filename, vocab_size, is_span)
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    if is_span:
        e.controversy_cie_span_train(e_config, data_loader, load_id)
    else:
        e.controversy_cie_train(e_config, data_loader, load_id)


def keyword_extract():
    e = ControversyExperiment()
    e.view_keyword()


def lm():
    e = ControversyExperiment()
    e.lm_baseline()


def lm_protest():
    e = ControversyExperiment()
    e.lm_protest_baseline()
    e.lm_protext_ex()


def protest_bert():
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    e_config = ExperimentConfig()
    e_config.name = "protest"
    e_config.num_epoch = 1
    e_config.save_interval = 1 * 60  # 1 minutes
    e_config.load_names = ['bert']
    vocab_size = 30522
    vocab_filename = "bert_voca.txt"

    data_loader = protest.DataLoader(hp.seq_max, vocab_filename, vocab_size)
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    e.train_protest(e_config, data_loader, load_id)

def wikicont_bert():
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    e_config = ExperimentConfig()
    e_config.name = "WikiContrv"
    e_config.num_epoch = 1
    e_config.save_interval = 60 * 60  # 1 minutes
    e_config.load_names = ['bert']
    e_config.valid_freq = 100
    vocab_size = 30522
    vocab_filename = "bert_voca.txt"

    data_loader = Ams18.DataLoader(hp.seq_max, vocab_filename, vocab_size)
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    e.train_wiki_contrv(e_config, data_loader, load_id)



def eval_all_contrv():
    return evals.controversy.eval_all_contrv()

def get_tf10():
    e = ControversyExperiment()
    e.get_tf_10()



def train_agree():
    hp = hyperparams.HPBert()

    e_config = ExperimentConfig()
    e_config.num_epoch = 2
    e_config.save_interval = 100 * 60  # 30 minutes
    e_config.voca_size = 30522
    e_config.load_names = ['bert']
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    exp_purpose = "(dis)agree train"

    e = Experiment(hp)
    print(exp_purpose)
    e_config.name = "AgreeTrain"
    vocab_filename = "bert_voca.txt"
    data_loader = agree.DataLoader(hp.seq_max, vocab_filename)
    save_path = e.train_agree(e_config, data_loader, load_id)
    print(exp_purpose)

if __name__ == '__main__':
    action = "eval_all_contrv"
    locals()[action]()

