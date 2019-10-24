from models.transformer import hyperparams
from trainer.experiment import Experiment
from trainer.ExperimentConfig import ExperimentConfig
import time
from misc_lib import average
from trainer import loader
import sys
from data_generator.rte import rte
from google import gsutil
from data_generator.shared_setting import NLI
from trainer.tf_module import *
from data_generator.NLI import nli
from data_generator.adhoc import ws


def train_rte():
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    vocab_filename = "bert_voca.txt"
    #load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    load_id = ("tlm_simple","model.ckpt-15000")

    e_config = ExperimentConfig()
    e_config.name = "RTE_{}".format("tlm_simple_15000")
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



def train_nil_on_bert():
    print('train_nil_on_bert')
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"
    e_config = ExperimentConfig()
    e_config.name = "NLI_10k_bert_cold"
    e_config.num_epoch = 2
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert']  # , 'aux_conflict']

    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    dir_name = "bert_cold"
    model_step =10 * 1000
    load_id = (dir_name, "model.ckpt-{}".format(model_step))
    print(load_id)
    saved = e.train_nli_ex_0(nli_setting, e_config, data_loader, load_id, False)
    e.test_acc2(nli_setting, e_config, data_loader, saved)


def test_nli():
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"
    e_config = ExperimentConfig()
    e_config.name = "NLI_400k_tlm_simple_wo_hint"
    e_config.num_epoch = 2
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert', 'cls_dense']  # , 'aux_conflict']
    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    #saved = "/mnt/scratch/youngwookim/Chair/output/model/runs/NLI_Cold/model-0"
    saved = "/mnt/scratch/youngwookim/Chair/output/model/runs/NLI_400k_tlm_wo_hint/model-0"
    saved = '/mnt/scratch/youngwookim/Chair/output/model/runs/NLI_400k_tlm_simple_hint/model-0'
    print(saved)
    e.test_acc2(nli_setting, e_config, data_loader, saved)


def train_nil_cold():
    print('train_nil_cold')
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"
    e_config = ExperimentConfig()
    e_config.name = "NLI_Cold"
    e_config.num_epoch = 2
    e_config.save_interval = 30 * 60  # 30 minutes

    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    saved = e.train_nli_ex_0(nli_setting, e_config, data_loader, None, False)
    e.test_acc2(nli_setting, e_config, data_loader, saved)



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
    r = average(scores)
    print("Avg\n{0:.03f}".format(r))
    return r


def fetch_bert(model_step):
    dir_path = "gs://clovertpu/training/model/tlm1_local"
    save_name = "tlm1_local_{}".format(model_step)
    load_id = gsutil.download_model(dir_path, model_step, save_name)
    return load_id


def download_and_run():
    for model_step in [5000, 10000,15000]:
        load_id = fetch_bert(model_step)
        train_test_repeat(load_id, "tlm1_{}".format(model_step), 10)

def tlm_test(dir_name):
    summary = {}
    k = 1000
    for model_step in [20*k]:
        load_id = (dir_name, "model.ckpt-{}".format(model_step))
        r = train_test_repeat(load_id, "{}_{}".format(dir_name, model_step), 5)
        print(load_id, r)
        summary[load_id] = r

    for key in summary:
        print(key, summary[key])


def bert_lm_test():
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
    e.bert_lm_pos_neg(e_config, data_loader, load_id)


def tlm_simple_cold_wo_hint():
    tlm_test("tlm_simple_wo_hint")

def tlm_simple_cold():
    tlm_test("tlm_simple_cold")

def tlm_simple_cold_256():
    tlm_test("tlm_simple_cold_256")


def tlm_simple_tune():
    tlm_test("tlm_simple_tune")

def tlm_bert_cold():
    tlm_test("bert_cold")

def baseline_run():
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    train_test_repeat(load_id, "bert_base", 10)


if __name__ == '__main__':
    begin = time.time()
    action = "bert_lm_test"
    locals()[action]()

    elapsed = time.time() - begin
    print("Total time :", elapsed)