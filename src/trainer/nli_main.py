import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


from task.transformer_est import Transformer, Classification
from models.transformer import bert
from models.transformer import hyperparams
from data_generator.shared_setting import NLI
from trainer.tf_module import *
import tensorflow as tf
from data_generator.NLI import nli
from trainer.experiment import Experiment
from trainer.ExperimentConfig import ExperimentConfig




def train_nil():
    hp = hyperparams.HPDefault()
    e = Experiment(hp)
    nli_setting = NLI()

    e_config = ExperimentConfig()
    e_config.name = "NLI_run_{}".format("A")
    e_config.num_epoch = 2
    e_config.save_interval = 30 * 60  # 30 minutes


    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename)
    e.train_nli(nli_setting, e_config, data_loader)



def train_nil_on_bert():
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"
    e_config = ExperimentConfig()
    e_config.name = "NLI_run_{}".format("A")
    e_config.num_epoch = 2
    e_config.save_interval = 30 * 60  # 30 minutes


    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    #load_id = None
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    #load_id = ("NLI_bert_w_explain", 'model-91531')
    #load_id = ("NLI_Only_A", "model-0")
    e.train_nli_ex_0(nli_setting, e_config, data_loader, load_id, False)



def train_nli_with_reinforce():
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"

    e_config = ExperimentConfig()
    e_config.name = "NLIEx_{}".format("H")
    e_config.num_epoch = 1
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert', 'cls_dense'] #, 'aux_conflict']

    explain_tag = 'conflict'  # 'dontcare'  'match' 'mismatch'

    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    #load_id = ("NLI_run_nli_warm", "model-97332")
    #load_id = ("NLIEx_A", "model-16910")
    #load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    #load_id = ("NLIEx_D", "model-1964")
    #load_id = ("NLIEx_D", "model-1317")
    load_id = ("NLI_run_A", 'model-0')
    e.train_nli_ex_1(nli_setting, e_config, data_loader, load_id, explain_tag)



def train_nli_with_premade():
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"

    e_config = ExperimentConfig()
    e_config.name = "NLIEx_{}".format("HB")
    e_config.num_epoch = 1
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert'] #, 'cls_dense'] #, 'aux_conflict']

    explain_tag = 'conflict'  # 'dontcare'  'match' 'mismatch'

    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    #load_id = ("NLI_run_nli_warm", "model-97332")
    #load_id = ("NLIEx_A", "model-16910")
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    e.train_nli_ex_with_premade_data(nli_setting, e_config, data_loader, load_id, explain_tag)



def analyze_nli_ex():
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"

    e_config = ExperimentConfig()
    e_config.name = "NLIEx_{}_align".format("E")
    e_config.num_epoch = 4
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert' ,'cls_dense', 'aux_conflict']


    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    load_id = ("NLIEx_E_align", "model-23621")
    e.nli_visualization(nli_setting, e_config, data_loader, load_id)


def baseline_explain():
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"

    e_config = ExperimentConfig()
    e_config.name = "NLI_run_{}".format("nli_warm")
    e_config.num_epoch = 4
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert', 'cls_dense']

    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    load_id = ("NLI_bert_w_explain", "model-91531")
    e.nli_explain_baselines(nli_setting, e_config, data_loader, load_id)



def attribution_explain():
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"

    e_config = ExperimentConfig()
    e_config.name = "NLI_run_{}".format("nli_eval")
    e_config.num_epoch = 4
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert', 'cls_dense']

    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    load_id = ("NLI_run_nli_warm", "model-97332")
    e.nli_attribution_baselines(nli_setting, e_config, data_loader, load_id)



def train_nli_with_reinforce_old():
    hp = hyperparams.HPNLI2()
    e = Experiment(hp)
    nli_setting = NLI()

    e_config = ExperimentConfig()
    e_config.name = "NLI_run_{}".format("retest")
    e_config.num_epoch = 4
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert', 'dense_cls'] #, 'aux_conflict']

    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename)
    load_id = ("interval", "model-48040")
    e.train_nli_ex_0(nli_setting, e_config, data_loader, load_id, True)


if __name__ == '__main__':
    action = "train_nli_with_premade"
    locals()[action]()
