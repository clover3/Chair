from models.transformer import hyperparams
from trainer.experiment import Experiment
from trainer.ExperimentConfig import ExperimentConfig
import time
from trainer import loader
import sys
from data_generator.rte import rte

def train_rte():
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    vocab_filename = "bert_voca.txt"
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')

    e_config = ExperimentConfig()
    e_config.name = "RTE_{}".format("A")
    e_config.num_epoch = 10
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert']

    data_loader = rte.DataLoader(hp.seq_max, vocab_filename, True)
    e.train_rte(e_config, data_loader, load_id)



def gradient_rte_visulize():
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    vocab_filename = "bert_voca.txt"
    load_id = loader.find_model_name("RTE_A")
    e_config = ExperimentConfig()
    e_config.name = "RTE_{}".format("visual")
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert', 'cls_dense']

    data_loader = rte.DataLoader(hp.seq_max, vocab_filename, True)
    e.rte_visualize(e_config, data_loader, load_id)




if __name__ == '__main__':
    begin = time.time()
    action = "gradient_rte_visulize"
    locals()[action]()

    elapsed = time.time() - begin
    print("Total time :", elapsed)