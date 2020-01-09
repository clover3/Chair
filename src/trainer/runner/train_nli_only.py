from data_generator.NLI import nli
from data_generator.shared_setting import NLI
from models.transformer import hyperparams
from trainer.ExperimentConfig import ExperimentConfig
from trainer.experiment import Experiment


def train_nil_on_bert():
    hp = hyperparams.HPBert()
    hp.batch_size = 16
    hp.lr = 3e-5
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"
    e_config = ExperimentConfig()
    e_config.name = "NLI_Only_{}".format("E")
    e_config.num_epoch = 3
    e_config.save_interval = 30 * 60  # 30 minutes

    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    e.train_nli_only_new(nli_setting, e_config, data_loader, load_id)

if __name__  == "__main__":
    train_nil_on_bert()