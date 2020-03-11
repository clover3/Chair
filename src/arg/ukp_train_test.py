import data_generator.argmining.ukp_header
from data_generator.NLI import nli
from data_generator.argmining.ukp import BertDataLoader
from data_generator.shared_data import SharedFeeder
from data_generator.shared_setting import NLI
from misc_lib import *
from models.transformer import hyperparams
from trainer.ExperimentConfig import ExperimentConfig
from trainer.experiment import Experiment


def ukp_train_test(load_id, exp_name):
    hp = hyperparams.HPBert()
    e_config = ExperimentConfig()
    e_config.num_epoch = 2
    e_config.save_interval = 100 * 60  # 30 minutes
    e_config.voca_size = 30522
    e_config.load_names = ['bert']
    encode_opt = "is_good"

    print(load_id)
    f1_list = []
    for topic in data_generator.argmining.ukp_header.all_topics:
        e = Experiment(hp)
        print(exp_name)
        e_config.name = "arg_{}_{}_{}".format(exp_name, topic, encode_opt)
        data_loader = BertDataLoader(topic, True, hp.seq_max, "bert_voca.txt", option=encode_opt)
        save_path = e.train_ukp(e_config, data_loader, load_id)
        print(topic)
        f1_last = e.eval_ukp(e_config, data_loader, save_path)
        f1_list.append((topic, f1_last))
    print(exp_name)
    print(encode_opt)
    print(f1_list)
    for key, score in f1_list:
        print("{0}\t{1:.03f}".format(key,score))


def ukp_train_test_repeat(load_id, exp_name, topic, n_repeat):
    hp = hyperparams.HPBert()
    e_config = ExperimentConfig()
    e_config.num_epoch = 2
    e_config.save_interval = 100 * 60  # 30 minutes
    e_config.voca_size = 30522
    e_config.load_names = ['bert']
    encode_opt = "is_good"

    print(load_id)
    scores = []
    for i in range(n_repeat):
        e = Experiment(hp)
        print(exp_name)
        e_config.name = "arg_{}_{}_{}".format(exp_name, topic, encode_opt)
        data_loader = BertDataLoader(topic, True, hp.seq_max, "bert_voca.txt", option=encode_opt)
        save_path = e.train_ukp(e_config, data_loader, load_id)
        f1_last = e.eval_ukp(e_config, data_loader, save_path)
        scores.append(f1_last)
    print(exp_name)
    print(encode_opt)
    for e in scores:
        print(e, end="\t")
    print()
    print("Avg\n{0:.03f}".format(average(scores)))


def train_ukp_with_nli(load_id, exp_name):
    step_per_epoch = 24544 + 970

    hp = hyperparams.HPBert()
    e_config = ExperimentConfig()
    e_config.num_steps = step_per_epoch
    e_config.save_interval = 100 * 60  # 30 minutes
    e_config.voca_size = 30522
    e_config.num_dev_batches = 30
    e_config.load_names = ['bert']
    e_config.valid_freq = 500
    encode_opt = "is_good"
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"
    num_class_list = [3, 3]
    f1_list = []
    for topic in data_generator.argmining.ukp_header.all_topics:
        e = Experiment(hp)
        print(exp_name)
        e_config.name = "argmix_{}_{}_{}".format(exp_name, topic, encode_opt)
        arg_data_loader = BertDataLoader(topic, True, hp.seq_max, "bert_voca.txt", option=encode_opt)
        nli_data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)

        shared_data_loader = SharedFeeder([arg_data_loader, nli_data_loader],
                                          [1, 5],
                                          ["Arg", "NLI"],
                                          hp.batch_size)

        save_path = e.train_shared(e_config, shared_data_loader, num_class_list, load_id)
        print(topic)
        f1_last = e.eval_ukp_on_shared(e_config, arg_data_loader, num_class_list, save_path)
        f1_list.append((topic, f1_last))
    print(exp_name)
    print(encode_opt)
    print(f1_list)
    for key, score in f1_list:
        print("{0}\t{1:.03f}".format(key,score))



def eval_ukp_with_nli(exp_name):
    step_per_epoch = 24544 + 970

    hp = hyperparams.HPBert()
    e_config = ExperimentConfig()
    e_config.num_steps = step_per_epoch
    e_config.voca_size = 30522
    e_config.num_dev_batches = 30
    e_config.load_names = ['bert']
    encode_opt = "is_good"
    num_class_list = [3, 3]
    f1_list = []
    save_path = "/mnt/scratch/youngwookim/Chair/output/model/runs/argmix_AN_B_40000_abortion_is_good/model-21306"
    for topic in data_generator.argmining.ukp_header.all_topics[:1]:
        e = Experiment(hp)
        print(exp_name)
        e_config.name = "argmix_{}_{}_{}".format(exp_name, topic, encode_opt)
        arg_data_loader = BertDataLoader(topic, True, hp.seq_max, "bert_voca.txt", option=encode_opt)
        f1_last = e.eval_ukp_on_shared(e_config, arg_data_loader, num_class_list, save_path)
        f1_list.append((topic, f1_last))
    print(exp_name)
    print(encode_opt)
    print(f1_list)
    for key, score in f1_list:
        print("{0}\t{1:.03f}".format(key,score))

