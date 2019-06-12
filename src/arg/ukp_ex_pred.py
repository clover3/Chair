from trainer.arg_experiment import ArgExperiment
from models.transformer import hyperparams
from trainer.experiment import Experiment
from trainer.ExperimentConfig import ExperimentConfig

from data_generator.argmining import ukp
from data_generator.argmining.ukp import BertDataLoader, PairedDataLoader, FeedbackData, NLIAsStance, SingleTopicLoader, StreamExplainer
from data_generator.argmining import NextSentPred, DocStance
from arg.ukp_train_test import *
import sys
from misc_lib import *


def run_dir():
    dir_path = "/mnt/scratch/youngwookim/data/clueweb12_10000_clean_sent"
    topic_idx = 3
    for file_path in get_dir_files(dir_path):
        run_ukp_ex(file_path, topic_idx)



def run_ukp_ex(file_path, topic_idx):
    hp = hyperparams.HPBert()
    e_config = ExperimentConfig()
    e_config.voca_size = 30522
    e_config.load_names = ['bert', 'cls_dense', 'aux_conflict']
    hp.batch_size = 512 + 512 - 128
    encode_opt = "is_good"
    is_3way = True
    explain_tag = 'polarity'
    print(file_path)
    #explain_tag = 'relevance'

    topic = ukp.all_topics[topic_idx]
    print("Blind : ", topic)
    e = Experiment(hp)
    e_config.name = "pred_arg_exp_{}_{}".format(topic, explain_tag)
    load_run_name = "arg_exp_{}_{}".format(topic, explain_tag)
    data_loader = StreamExplainer(topic, file_path, is_3way, hp.seq_max, "bert_voca.txt", option=encode_opt)
    e.run_ukp_ex(e_config, data_loader, load_run_name)



if __name__ == '__main__':
    begin = time.time()
    action = "run_dir"
    locals()[action]()
    elapsed = time.time() - begin
    print("Total time :", elapsed)