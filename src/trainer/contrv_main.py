import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


from task.transformer_est import Transformer, Classification
from models.transformer import bert
from models.transformer import hyperparams
from data_generator.shared_setting import NLI
from trainer.tf_module import *
import tensorflow as tf
from data_generator.adhoc import ws
from data_generator.controversy import mscore
from trainer.experiment import Experiment
from trainer.ExperimentConfig import ExperimentConfig



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


def test_mscore_eval():
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
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert', 'cls']
    vocab_size = 30522
    vocab_filename = "bert_voca.txt"

    data_loader = ws.DataLoader(hp.seq_max, vocab_filename, vocab_size)
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    e.controv_lm(e_config, data_loader, load_id)


if __name__ == '__main__':
    action = "train_mscore_regression"
    locals()[action]()
