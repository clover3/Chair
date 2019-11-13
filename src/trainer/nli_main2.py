
from data_generator.shared_setting import NLI
from trainer.tf_module import *
import tensorflow as tf
from data_generator.NLI import nli
from data_generator.ubuntu import ubuntu
from trainer.experiment import Experiment
from trainer.ExperimentConfig import ExperimentConfig
from models.transformer.hp_finetue import HP
from models.transformer import hyperparams
from tlm.data_gen.read_tfrecord import extract_stream

def train_nil():
    hp = HP()
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"

    e_config = ExperimentConfig()
    e_config.name = "NLI_only_{}".format("512")
    e_config.num_epoch = 2
    e_config.save_interval = 30 * 60  # 30 minutes


    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    e.train_nli_ex_0(nli_setting, e_config, data_loader, load_id, False)

def train_mnli_any_way():
    hp = HP()
    hp.batch_size = 8
    hp.compare_deletion_num = 20
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"
    e_config = ExperimentConfig()
    e_config.name = "NLIEx_Any_512"
    e_config.ex_val = False
    e_config.num_epoch = 1
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert', 'cls_dense']  # , 'aux_conflict']
    e_config.v2_load = True
    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    load_id = ("nli512", 'model.ckpt-65000')
    e.train_nli_any_way(nli_setting, e_config, data_loader, load_id)


def visualize_senli_on_plain_text():
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"

    e_config = ExperimentConfig()
    e_config.name = "NLIEx_lm_analyze"
    e_config.num_epoch = 4
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert' ,'cls_dense', 'aux_conflict']

    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    load_id = ("NLIEx_Any_A", "model-7255")

    p = "C:\work\Code\Chair\data\\tf\\0"
    data = extract_stream(p)
    e.nli_visualization_lm(nli_setting, e_config, data_loader, load_id, data)



if __name__ == '__main__':
    action = "visualize_senli_on_plain_text"
    locals()[action]()
