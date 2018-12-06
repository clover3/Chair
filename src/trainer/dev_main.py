import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


#print("LD_LIBRARY_PATH : {}".format(os.environ["LD_LIBRARY_PATH"]))

from trainer.experiment import Experiment
from trainer.ExperimentConfig import ExperimentConfig

from models.transformer.hyperparams import *
from data_generator import shared_setting
from data_generator.mask_lm import enwiki, guardian, tweets, author_as_doc
from data_generator.pair_lm import loader
from data_generator.stance import stance_detection
from data_generator.data_parser import tweet_reader
from data_generator.aux_pair.loader import AuxPairLoader

def lm_train():
    hp = Hyperparams()
    e_config = ExperimentConfig()
    e = Experiment(hp)
    e.train_lm_inf()



def lm_guardian_train():
    hp = Hyperparams()
    guardian_data = guardian.GuardianLoader("atheism", hp.seq_max, shared_setting.Guardian2Stance)
    e_config = ExperimentConfig()
    e_config.name = "LM_guardian"
    e_config.num_epoch = 30

    e = Experiment(hp)
    e.train_lm_batch(e_config, guardian_data)

def lm_tweets_train():
    hp = HPTweets()
    data = tweets.TweetLoader("atheism", hp.seq_max, shared_setting.Tweets2Stance)
    e_config = ExperimentConfig()
    e_config.name = "LM_tweets"
    e_config.num_epoch = 30
    e_config.save_interval = 30 * 60  # 30 minutes
    e = Experiment(hp)
    e.train_lm_batch(e_config, data)




def stance_after_lm():
    hp = HPFineTunePair()
    topic = "atheism"
    e = Experiment(hp)
    preload_id = ("DLM_pair_tweets_atheism", 840965)
    setting = shared_setting.TopicTweets2Stance(topic)
    stance_data = stance_detection.FineLoader(topic, hp.seq_max, setting.vocab_filename, hp.sent_max)
    e.train_stance(setting.vocab_size, stance_data, preload_id)


def stance_fine_tune():
    # importing the required module
    import matplotlib.pyplot as plt


    for lr in [1e-3, 5e-4, 2e-4,1e-4]:
        hp = HPFineTunePair()
        topic = "hillary"
        e = Experiment(hp)
        hp.lr = lr
        hp.num_epochs = 100
        preload_id = ("LM_pair_tweets_hillary_run2", 1247707)
        setting = shared_setting.TopicTweets2Stance(topic)
        stance_data = stance_detection.FineLoader(topic, hp.seq_max, setting.vocab_filename, hp.sent_max)
        valid_history = e.train_stance(setting.vocab_size, stance_data, preload_id)
        e.clear_run()

        l_acc, l_f1 = zip(*valid_history)
        plt.plot(l_acc, label="{} / ACC".format(lr))
        plt.plot(l_f1, label="{} / F1".format(lr))

    plt.legend(loc='lower right')

    # giving a title to my graph
    plt.title('learning rate - dev !')
    # function to show the plot
    plt.show()


def stance_after_feature():
    hp = HPPairFeatureTweetFine()
    topic = "atheism"
    e = Experiment(hp)
    preload_id = ("LM_pair_featuer_tweets_atheism", 979)
    setting = shared_setting.TopicTweets2Stance(topic)
    stance_data = stance_detection.DataLoader(topic, hp.seq_max, setting.vocab_filename)
    e.train_stance_pair_feature(setting.vocab_size, stance_data, None)



def pair_lm():
    hp = HPPairTweet()
    topic = "atheism"
    setting = shared_setting.TopicTweets2Stance(topic)
    tweet_group = tweet_reader.load_per_user(topic)
    data = loader.PairDataLoader(hp.sent_max, setting, tweet_group)
    e_config = ExperimentConfig()
    e_config.name = "LM_pair_tweets_{}".format(topic)
    e_config.num_epoch = 1
    e_config.save_interval = 30 * 60  # 30 minutes
    e = Experiment(hp)
    e.train_pair_lm(e_config, data)


def document_lm():
    hp = HPDocLM()
    topic = "atheism"
    setting = shared_setting.TopicTweets2Stance(topic)
    use_cache = True


    run_id = "{}_{}".format(topic, hp.seq_max)
    if use_cache:
        data = author_as_doc.AuthorAsDoc.load_from_pickle(run_id)
    else:
        tweet_group = tweet_reader.load_per_user(topic)
        data = author_as_doc.AuthorAsDoc(hp.seq_max, setting, tweet_group)
        data.index_data()
        data.save_to_pickle(run_id)

    e_config = ExperimentConfig()
    e_config.name = "DLM_pair_tweets_{}".format(topic)
    e_config.num_epoch = 1
    e_config.save_interval = 30 * 60  # 30 minutes
    e = Experiment(hp)
    e.train_doc_lm(e_config, data)


def pair_lm_inf():
    hp = HPPairTweet()
    topic = "atheism"
    setting = shared_setting.TopicTweets2Stance(topic)
    use_cache = False
    run_id = "{}_{}".format(topic, hp.sent_max)
    if use_cache :
        print("using PairDataCache")
        data = loader.PairDataLoader.load_from_pickle(run_id)
    else:
        tweet_group = tweet_reader.load_per_user(topic)
        data = loader.PairDataLoader(hp.sent_max, setting, tweet_group)
        data.index_data()
        data.save_to_pickle(run_id)

    e_config = ExperimentConfig()
    e_config.name = "LM_pair_tweets_{}".format(topic)
    e_config.num_epoch = 1
    e_config.save_interval = 30 * 60  # 30 minutes
    e = Experiment(hp)
    e.train_pair_lm_inf(e_config, data)


def pair_feature():
    hp = HPPairFeatureTweet()
    topic = "atheism"
    setting = shared_setting.TopicTweets2Stance(topic)
    use_cache = True
    run_id = "{}_{}".format(topic, hp.sent_max)
    if use_cache :
        print("using PairDataCache")
        data = loader.PairDataLoader.load_from_pickle(run_id)
    else:
        tweet_group = tweet_reader.load_per_user(topic)
        data = loader.PairDataLoader(hp.sent_max, setting, tweet_group)
        data.index_data()
        data.save_to_pickle(run_id)

    e_config = ExperimentConfig()
    e_config.name = "LM_pair_featuer_tweets_{}".format(topic)
    e_config.num_epoch = 1
    e_config.save_interval = 3 * 60  # 3 minutes
    e = Experiment(hp)
    e.train_pair_feature(e_config, data)



def stance_cold_start():
    hp = HPColdStart()
    e = Experiment(hp)
    topic = "hillary"
    setting = shared_setting.TopicTweets2Stance(topic)
    stance_data = stance_detection.DataLoader(topic, hp.seq_max, setting.vocab_filename)

    voca_size = setting.vocab_size
    e.train_stance(voca_size, stance_data)


def stance_with_consistency():
    hp = HPStanceConsistency()
    topic = "atheism"
    e = Experiment(hp)
    e_config = ExperimentConfig()
    e_config.name = "stance_consistency_{}".format(topic)

    setting = shared_setting.TopicTweets2Stance(topic)
    stance_data = stance_detection.DataLoader(topic, hp.seq_max, setting.vocab_filename)
    tweet_group = tweet_reader.load_per_user(topic)
    aux_data = AuxPairLoader(hp.seq_max, setting, tweet_group)
    voca_size = setting.vocab_size
    e.train_stance_consistency(voca_size, stance_data, aux_data)

def baselines():
    hp = Hyperparams()
    e = Experiment(hp)
    e.stance_baseline()


if __name__ == '__main__':
    action = "stance_after_lm"
    locals()[action]()
