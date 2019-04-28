from trainer.arg_experiment import ArgExperiment
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from models.transformer import hyperparams
from trainer.experiment import Experiment
from trainer.ExperimentConfig import ExperimentConfig

from data_generator.argmining import ukp
from data_generator.argmining.ukp import BertDataLoader


def uni_lm():
    e = ArgExperiment()
    e.train_lr_3way()



def kl_test():
    e = ArgExperiment()
    e.train_lr_2way()


def inspect():
    e = ArgExperiment()
    #e.tf_stat()
    e.divergence()


def lr_kl():
    e = ArgExperiment()
    #e.tf_stat()
    e.divergence_lr()


def summarize():
    e = ArgExperiment()
    e.summarize()


def train_bert():
    hp = hyperparams.HPBert()

    e_config = ExperimentConfig()
    e_config.num_epoch = 2
    e_config.save_interval = 1 * 60  # 30 minutes
    e_config.voca_size = 30522
    e_config.load_names = ['bert']
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    load_id = ("NLI_Only_B", 'model-0')

    f1_list = []
    for topic in ukp.all_topics:
        e = Experiment(hp)
        print("BERT")
        e_config.name = "arg_key_neccesary_{}".format(topic)
        data_loader = BertDataLoader(topic, True, hp.seq_max, "bert_voca.txt")
        f1_last = e.train_ukp(e_config, data_loader, load_id)
        print(topic)
        print(f1_last)
        f1_list.append((topic, f1_last))
    print("BERT")
    print(f1_list)


def failure_analysis():
    for topic in ukp.all_topics:
        hp = hyperparams.HPBert()
        e = Experiment(hp)
        e_config = ExperimentConfig()
        e_config.voca_size = 30522
        e_config.load_names = ['bert']
        e_config.name = "arg_b_{}".format(topic)
        e.failure_ukp(e_config, topic)


if __name__ == '__main__':
    action = "failure_analysis"
    locals()[action]()