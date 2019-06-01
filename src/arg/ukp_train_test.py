from models.transformer import hyperparams
from trainer.experiment import Experiment
from trainer.ExperimentConfig import ExperimentConfig
from data_generator.argmining import ukp
from data_generator.argmining.ukp import BertDataLoader, PairedDataLoader, FeedbackData, NLIAsStance


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
    for topic in ukp.all_topics:
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
