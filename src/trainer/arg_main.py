from trainer.arg_experiment import ArgExperiment
from models.transformer import hyperparams
from trainer.experiment import Experiment
from trainer.ExperimentConfig import ExperimentConfig

from data_generator.argmining import ukp
from data_generator.argmining.ukp import BertDataLoader, PairedDataLoader, FeedbackData, NLIAsStance
from data_generator.argmining import NextSentPred
from trainer import loader
from arg.ukp_train_test import ukp_train_test
from google import gsutil

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
    #load_id = ("BERT_rel_small", 'model.ckpt-17000')
    #load_id = ("causal", 'model.ckpt-1000')
    #exp_purpose = "nli - way(2 epoch)"
    exp_purpose = "NLI_reverse"
    encode_opt = "only_topic_word_reverse"

    print(load_id)
    f1_list = []
    for topic in ukp.all_topics:
        e = Experiment(hp)
        print(exp_purpose)
        e_config.name = "arg_nli_{}_{}".format(topic, encode_opt)
        data_loader = BertDataLoader(topic, True, hp.seq_max, "bert_voca.txt", option=encode_opt)
        save_path = e.train_ukp(e_config, data_loader, load_id)
        print(topic)
        f1_last = e.eval_ukp(e_config, data_loader, save_path)
        f1_list.append((topic, f1_last))
    print(exp_purpose)
    print(encode_opt)
    print(f1_list)
    for key, score in f1_list:
        print("{0}\t{1:.03f}".format(key,score))


def test_model():
    hp = hyperparams.HPBert()

    e_config = ExperimentConfig()
    e_config.num_epoch = 2
    e_config.save_interval = 100 * 60  # 30 minutes
    e_config.voca_size = 30522
    e_config.load_names = ['bert']
    exp_purpose = "NLI_cheat"
    encode_opt = "is_good"

    f1_list = []
    for topic in ukp.all_topics:
        e = Experiment(hp)
        print(exp_purpose)
        e_config.name = "arg_nli_{}_{}".format(topic, encode_opt)

        cheat_topic = "cloning" if topic != "cloning" else "death_penalty"
        load_model_name = "arg_nli_{}_{}".format(cheat_topic, encode_opt)
        data_loader = BertDataLoader(topic, True, hp.seq_max, "bert_voca.txt", option=encode_opt)
        print(topic)
        save_path = loader.find_available(load_model_name)
        f1_last = e.eval_ukp(e_config, data_loader, save_path)
        f1_list.append((topic, f1_last))
    print(exp_purpose)
    print(encode_opt)
    print(f1_list)
    for key, score in f1_list:
        print("{0}\t{1:.03f}".format(key, score))

def train_ukp_ex():
    hp = hyperparams.HPBert()

    e_config = ExperimentConfig()
    e_config.num_epoch = 2
    e_config.save_interval = 5 * 60  # 30 minutes
    e_config.voca_size = 30522
    e_config.load_names = ['bert', 'dense_cls']
    exp_purpose = "ex"
    encode_opt = "is_good"
    is_3way = True
    f1_list = []
    explain_tag = 'polarity'
    explain_tag = 'relevance'

    for topic in ukp.all_topics[2:]:
        e = Experiment(hp)
        print(exp_purpose)
        e_config.name = "arg_exp_{}_{}".format(topic, explain_tag)
        load_run_name = "arg_nli_{}_is_good".format(topic)
        data_loader = BertDataLoader(topic, is_3way, hp.seq_max, "bert_voca.txt", option=encode_opt)
        save_path = e.train_ukp_ex(e_config, data_loader, load_run_name, explain_tag)

    print(exp_purpose)
    print(encode_opt)
    print(f1_list)
    for key, score in f1_list:
        print("{0}\t{1:.03f}".format(key, score))


def train_weighted():
    hp = hyperparams.HPBert()

    e_config = ExperimentConfig()
    e_config.num_epoch = 2
    e_config.save_interval = 100 * 60  # 30 minutes
    e_config.voca_size = 30522
    e_config.load_names = ['bert']
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    load_id = ("NLI_Only_B", 'model-0')
    #load_id = ("causal", 'model.ckpt-1000')
    exp_purpose = "Weighted 0.0 (2 epoch)"
    print(load_id)
    f1_list = []
    for topic in ukp.all_topics:
        e = Experiment(hp)
        print(exp_purpose)
        e_config.name = "arg_weight_{}".format(topic)
        data_loader = BertDataLoader(topic, True, hp.seq_max, "bert_voca.txt", option="weighted")
        data_loader.weight = 0.0
        save_path = e.train_ukp(e_config, data_loader, load_id)
        print(topic)
        f1_last = e.eval_ukp(e_config, data_loader, save_path)
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
    e_config.load_names = ['bert']
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
    e_config.load_names = ['bert', 'cls_dense']
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
        e_config.load_names = ['bert', 'cls_dense']
        e_config.name = "arg_nli_{}_is_good".format(topic)
        e.failure_ukp(e_config, topic)




def train_psf():
    for topic in ukp.all_topics[:1]:
        hp = hyperparams.HPBert()
        hp.lr = 1e-5
        e = Experiment(hp)
        e_config = ExperimentConfig()
        e_config.voca_size = 30522
        e_config.name = "arg_psf_{}".format(topic)
        e_config.preload_name = "arg_b_{}".format(topic)
        e_config.num_epochs = 2
        e_config.load_names = ['bert']
        e_config.preload_id = ("NLI_Only_B", 'model-0')
        feedback_data = FeedbackData(topic, True, hp.seq_max, "bert_voca.txt", "only_topic_word")
        e.pseudo_stance_ukp(e_config, topic, feedback_data)

def train_psf_vector():
    for topic in ukp.all_topics[:1]:
        hp = hyperparams.HPUKPVector()
        hp.lr = 1e-5
        hp.use_reorder = True
        e = Experiment(hp)
        e_config = ExperimentConfig()
        e_config.voca_size = 30522
        e_config.name = "arg_psf_{}".format(topic)
        e_config.preload_name = "arg_b_{}".format(topic)
        e_config.num_epochs = 200
        e_config.load_names = ['bert', 'cls_dense']
        e_config.preload_id = ("NLI_Only_B", 'model-0')
        feedback_data = FeedbackData(topic, True, hp.seq_max, "bert_voca.txt", "only_topic_word")
        e.pseudo_stance_ukp_vector(e_config, topic, feedback_data)



def train_topic_vector():
    for topic in ukp.all_topics[:1]:
        hp = hyperparams.HPUKPVector()
        hp.lr = 1e-5
        e = Experiment(hp)
        e_config = ExperimentConfig()
        e_config.voca_size = 30522
        e_config.name = "arg_topic_vector_{}".format(topic)
        e_config.preload_name = "arg_nli_{}_only_topic_word_reverse".format(topic)
        e_config.num_epochs = 200
        e_config.load_names = ['bert', 'cls_dense']
        feedback_data = BertDataLoader(topic, True, hp.seq_max, "bert_voca.txt", "only_topic_word")
        e.train_topic_vector(e_config, topic, feedback_data)


def test_nli_as_stance():
    for topic in ukp.all_topics[:1]:
        hp = hyperparams.HPUKPVector()
        e = Experiment(hp)
        e_config = ExperimentConfig()
        e_config.voca_size = 30522
        e_config.name = "arg_psf_{}".format(topic)
        e_config.preload_name = "arg_b_{}".format(topic)
        e_config.num_epochs = 200
        e_config.load_names = ['bert', 'cls_dense']
        e_config.preload_id = ("NLI_Only_B", 'model-0')
        data_loader = NLIAsStance(topic, True, hp.seq_max, "bert_voca.txt", option="only_topic_word")
        e.nli_as_stance(e_config, topic, data_loader)



def query_expansion():
    for topic in ukp.all_topics:
        hp = hyperparams.HPBert()
        e = Experiment(hp)
        e_config = ExperimentConfig()
        e_config.voca_size = 30522
        e_config.load_names = ['bert']
        e_config.name = "arg_b_{}".format(topic)
        e.pred_ukp_aux(e_config, topic)


def fetch_bert_and_train():
    model_step = 100000
    dir_path = "gs://clovertpu/training/model_abortionB"
    save_name = "Abortion_B"
    load_id = gsutil.download_model(dir_path, model_step, save_name)
    ukp_train_test(load_id, "bert_retrain")


if __name__ == '__main__':
    action = "fetch_bert_and_train"
    locals()[action]()
