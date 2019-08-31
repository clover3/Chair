
import sys
from task.transformer_est import Transformer, Classification
from models.transformer import bert
from models.transformer import hyperparams
from data_generator.shared_setting import NLI
from trainer.tf_module import *
import tensorflow as tf
from data_generator.NLI import nli
from data_generator.ubuntu import ubuntu
from trainer.experiment import Experiment
from trainer.ExperimentConfig import ExperimentConfig



def train_nli_smart_rf():
    hp = hyperparams.HPSENLI()
    hp.compare_deletion_num = 20
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"

    e_config = ExperimentConfig()

    #explain_tag = 'mismatch'
    explain_tag = 'match'
    #explain_tag = 'mismatch'

    loss_type = 2
    e_config.name = "NLIEx_Hinge_{}".format(explain_tag)
    e_config.num_epoch = 1
    e_config.ex_val = True
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert', 'cls_dense'] #, 'aux_conflict']
    e_config.save_eval = True
    e_config.save_name = "LossFn_{}_{}".format(loss_type, explain_tag)

    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    load_id = ("NLI_run_A", 'model-0')

    print("Loss : ", loss_type)

    e.train_nli_smart(nli_setting, e_config, data_loader, load_id, explain_tag, loss_type)


if __name__ == '__main__':
    action = "train_nli_smart_rf"
    locals()[action]()
