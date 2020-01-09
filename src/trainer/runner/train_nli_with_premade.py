import sys

from data_generator.NLI import nli
from data_generator.shared_setting import NLI
from models.transformer import hyperparams
from trainer.ExperimentConfig import ExperimentConfig
from trainer.experiment import Experiment


def train_nli_with_premade(explain_tag):
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"

    e_config = ExperimentConfig()
    e_config.name = "NLIEx_{}".format("Premade_"+explain_tag)
    e_config.num_epoch = 1
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert'] #, 'cls_dense'] #, 'aux_conflict']

    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    e.train_nli_ex_with_premade_data(nli_setting, e_config, data_loader, load_id, explain_tag)


if __name__  == "__main__":
    train_nli_with_premade(sys.argv[1])