from models.transformer import hyperparams
from trainer.experiment import Experiment
from trainer.ExperimentConfig import ExperimentConfig
import time
from misc_lib import average
from trainer import loader
import sys
from data_generator.rte import rte
from google import gsutil

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


def train_test_repeat(load_id, exp_name, n_repeat):
    hp = hyperparams.HPBert()
    e_config = ExperimentConfig()
    e_config.name = "RTE_{}".format("A")
    e_config.num_epoch = 10
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert']
    vocab_filename = "bert_voca.txt"
    data_loader = rte.DataLoader(hp.seq_max, vocab_filename, True)

    print(load_id)
    scores = []
    for i in range(n_repeat):
        e = Experiment(hp)
        print(exp_name)
        e_config.name = "rte_{}".format(exp_name)
        save_path = e.train_rte(e_config, data_loader, load_id)
        acc = e.eval_rte(e_config, data_loader, save_path)
        scores.append(acc)
    print(exp_name)
    for e in scores:
        print(e, end="\t")
    print()
    print("Avg\n{0:.03f}".format(average(scores)))


def fetch_bert(model_step):
    dir_path = "gs://clovertpu/training/model/tlm1_local"
    save_name = "tlm1_local_{}".format(model_step)
    load_id = gsutil.download_model(dir_path, model_step, save_name)
    return load_id


def download_and_run():
    for model_step in [5000, 10000,15000]:
        load_id = fetch_bert(model_step)
        train_test_repeat(load_id, "tlm1_{}".format(model_step), 10)

def baseline_run():
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    train_test_repeat(load_id, "bert_base", 10)


if __name__ == '__main__':
    begin = time.time()
    action = "download_and_run"
    locals()[action]()

    elapsed = time.time() - begin
    print("Total time :", elapsed)