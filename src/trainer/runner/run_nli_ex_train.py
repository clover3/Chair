import logging
import sys

from data_generator.NLI import nli
from data_generator.shared_setting import NLI
from models.transformer import hyperparams
from trainer.ExperimentConfig import ExperimentConfig
from trainer.experiment import Experiment


def train_nli_smart_rf(explain_tag):
    hp = hyperparams.HPSENLI()
    hp.compare_deletion_num = 20
    e = Experiment(hp)
    e.log.setLevel(logging.WARNING)
    e.log2.setLevel(logging.WARNING)
    e.log.info("I don't want to see")
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"

    e_config = ExperimentConfig()
    e_config.name = "NLIEx_{}".format("CO_"+explain_tag)
    e_config.num_epoch = 1
    e_config.ex_val = False
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert', 'cls_dense'] #, 'aux_conflict']
    e_config.save_payload = True

    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    load_id = ("NLI_run_A", 'model-0')
    e.train_nli_smart(nli_setting, e_config, data_loader, load_id, explain_tag, 5)



if __name__  == "__main__":
    train_nli_smart_rf(sys.argv[1])