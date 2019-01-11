import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from task.transformer_est import Transformer, Classification
from models.transformer import bert
from models.transformer import hyperparams
from data_generator.shared_setting import NLI
from trainer.tf_module import *
import tensorflow as tf
from data_generator.adhoc import ws
from trainer.experiment import Experiment
from trainer.ExperimentConfig import ExperimentConfig



def train_adhoc_with_reinforce():
    hp = hyperparams.HPAdhoc()
    e = Experiment(hp)

    e_config = ExperimentConfig()
    e_config.name = "Adhoc_{}".format("E")
    e_config.num_epoch = 4
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert']
    vocab_size = 30522
    vocab_filename = "bert_voca.txt"

    data_loader = ws.DataLoader(hp.seq_max, vocab_filename, vocab_size)
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    e.train_adhoc(e_config, data_loader, load_id)



def run_adhoc_rank():
    hp = hyperparams.HPAdhoc()
    hp.batch_size = 512

    e = Experiment(hp)

    e_config = ExperimentConfig()
    e_config.name = "Adhoc_{}_eval2".format("E")
    e_config.num_epoch = 4
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert', 'reg_dense']
    vocab_size = 30522
    vocab_filename = "bert_voca.txt"

    data_loader = ws.DataLoader(hp.seq_max, vocab_filename, vocab_size)
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    load_id = ("Adhoc_E", 'model-58338')
    e.rank_adhoc(e_config, data_loader, load_id)


def run_ql_rank():
    hp = hyperparams.HPQL()

    e = Experiment(hp)

    e_config = ExperimentConfig()
    e_config.name = "Adhoc_{}".format("D")
    e_config.num_epoch = 4
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert', 'cls']
    vocab_size = 30522
    vocab_filename = "bert_voca.txt"

    data_loader = ws.DataLoader(hp.seq_max, vocab_filename, vocab_size)
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    e.rank_ql(e_config, data_loader, load_id)


def test_ql():
    hp = hyperparams.HPAdhoc()
    hp.batch_size = 512

    e = Experiment(hp)

    e_config = ExperimentConfig()
    e_config.name = "Adhoc_{}".format("C")
    e_config.num_epoch = 4
    e_config.load_names = ['bert', 'cls']
    vocab_size = 30522
    vocab_filename = "bert_voca.txt"

    data_loader = ws.DataLoader(hp.seq_max, vocab_filename, vocab_size)
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    e.test_ql(e_config, data_loader, load_id)




if __name__ == '__main__':
    action = "run_adhoc_rank"
    locals()[action]()
