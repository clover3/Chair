from models.transformer import bert
from models.transformer import hyperparams
from data_generator.shared_setting import NLI
from trainer.tf_module import *
import tensorflow as tf
from data_generator.crs.claim_text_classifier import DataGenerator
from trainer.experiment import Experiment
from trainer.ExperimentConfig import ExperimentConfig
from data_generator.adhoc import score_loader
import path
from data_generator.adhoc.data_sampler import *
import sys



def crs_baseline():
    hp = hyperparams.HPCRS()
    hp.batch_size = 16 
    e = Experiment(hp)

    e_config = ExperimentConfig()
    e_config.name = "CRS_{}".format("baseline")
    e_config.num_epoch = 4
    e_config.save_interval = 10 * 60  # 60 minutes
    e_config.load_names = ['bert'] #, 'reg_dense']
    e_config.voca_size = 30522

    data_loader = DataGenerator()
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    e.train_crs_classify(e_config, data_loader, load_id)


if __name__ == '__main__':
    action = sys.argv[1]
    locals()[action]()
