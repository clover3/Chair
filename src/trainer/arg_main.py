from trainer.arg_experiment import ArgExperiment
from models.transformer import hyperparams
from trainer.experiment import Experiment
from trainer.ExperimentConfig import ExperimentConfig

from data_generator.argmining import ukp
from data_generator.argmining.ukp import BertDataLoader, PairedDataLoader
from data_generator.argmining import NextSentPred


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
    e_config.save_interval = 100 * 60  # 30 minutes
    e_config.voca_size = 30522
    e_config.load_names = ['bert']
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    load_id = ("NLI_Only_B", 'model-0')
    #load_id = ("causal", 'model.ckpt-1000')
    exp_purpose = "keyword expansion, no 'good' (2 epoch)"

    f1_list = []
    for topic in ukp.all_topics:
        e = Experiment(hp)
        print(exp_purpose)
        e_config.name = "arg_ma_ex_{}".format(topic)
        data_loader = BertDataLoader(topic, True, hp.seq_max, "bert_voca.txt")
        f1_last = e.train_ukp(e_config, data_loader, load_id)
        print(topic)
        print(f1_last)
        f1_list.append((topic, f1_last))
    print(exp_purpose)
    print(f1_list)
    for key, score in f1_list:
        print("{0}\t{1:.03f}".format(key,score))



def train_next_pred():
    hp = hyperparams.HPBert()

    e_config = ExperimentConfig()
    e_config.num_epoch = 2
    e_config.save_interval = 100 * 60  # 30 minutes
    e_config.voca_size = 30522
    e_config.load_names = ['bert', 'dense_cls']
    #load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    #load_id = ("NLI_Only_B", 'model-0')
    #load_id = ("causal", 'model.ckpt-1000')
    exp_purpose = "Next pred"

    for topic in ukp.all_topics[2:3]:
        e = Experiment(hp)
        print(exp_purpose)
        print(topic)
        e_config.name = "arg_nextpred_{}".format(topic)
        load_name = "arg_b_{}".format(topic)
        data_loader1 = BertDataLoader(topic, True, hp.seq_max, "bert_voca.txt")
        data_loader2 = NextSentPred.DataLoader(topic, hp.seq_max)
        e.train_next_sent(e_config, data_loader1, data_loader2, load_name)


def train_concat():
    hp = hyperparams.HPBert()

    e_config = ExperimentConfig()
    e_config.num_epoch = 2
    e_config.save_interval = 1 * 60  # 30 minutes
    e_config.voca_size = 30522
    e_config.load_names = ['bert']
    load_id1 = ("NLI_Only_B", 'model-0')
    load_id2 = ("causal", 'model.ckpt-1000')

    f1_list = []
    for topic in ukp.all_topics:
        e = Experiment(hp)
        print("train_concat")
        e_config.name = "arg_concat_{}".format(topic)
        data_loader = BertDataLoader(topic, True, hp.seq_max, "bert_voca.txt")
        f1_last = e.train_ukp_concat(e_config, data_loader, load_id1, load_id2)
        print(topic)
        print(f1_last)
        f1_list.append((topic, f1_last))
    print("train_concat")
    print(f1_list)



def train_paired():
    hp = hyperparams.HPBert()
    hp.batch_size = 16
    e_config = ExperimentConfig()
    e_config.num_epoch = 4
    e_config.save_interval = 1 * 60  # 30 minutes
    e_config.voca_size = 30522
    e_config.load_names = ['bert']
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    load_id = ("NLI_Only_B", 'model-0')

    f1_list = []
    for topic in ukp.all_topics:
        e = Experiment(hp)
        print(load_id[0])
        e_config.name = "arg_paired_{}".format(topic)
        data_loader = PairedDataLoader(topic, True, hp.seq_max, "bert_voca.txt")
        f1_last = e.train_ukp_paired(e_config, data_loader, load_id)
        print(topic)
        print(f1_last)
        f1_list.append((topic, f1_last))
    print("Seg 3 No Pair")
    print(f1_list)
    for key, score in f1_list:
        print("{}\t{}".format(key,score))


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
    action = "train_next_pred"
    locals()[action]()