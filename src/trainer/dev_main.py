import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


#print("LD_LIBRARY_PATH : {}".format(os.environ["LD_LIBRARY_PATH"]))

from trainer.experiment import Experiment
from trainer.ExperimentConfig import ExperimentConfig

from models.transformer.hyperparams import *
from data_generator import shared_setting
from data_generator.mask_lm import enwiki, guardian, tweets
from data_generator.pair_lm import loader
from data_generator.stance import stance_detection
from data_generator.data_parser import tweet_reader

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
    preload_id = ("LM_pair_tweets_atheism", 20934)
    setting = shared_setting.TopicTweets2Stance(topic)
    stance_data = stance_detection.DataLoader(topic, hp.seq_max, setting.vocab_filename)
    e.train_stance(setting.vocab_size, stance_data, preload_id)


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


def pair_lm_inf():
    hp = HPPairTweet()
    topic = "atheism"
    setting = shared_setting.TopicTweets2Stance(topic)
    tweet_group = tweet_reader.load_per_user(topic)
    data = loader.PairDataLoader(hp.sent_max, setting, tweet_group)
    e_config = ExperimentConfig()
    e_config.name = "LM_pair_tweets_debug_{}".format(topic)
    e_config.num_epoch = 1
    e_config.save_interval = 30 * 60  # 30 minutes
    e = Experiment(hp)
    e.train_pair_lm_inf(e_config, data)

def stance_after_guardian_lm():
    hp = HPFineTune()
    e = Experiment(hp)
    preload_id = ("LM_guardian", 25911)
    voca_size = shared_setting.Enwiki2Stance.vocab_size
    e.train_stance(voca_size, preload_id)



def stance_cold_start():
    hp = Hyperparams()
    e = Experiment(hp)
    voca_size = shared_setting.Enwiki2Stance.vocab_size
    e.train_stance(voca_size)


def baselines():
    hp = Hyperparams()
    e = Experiment(hp)
    e.stance_baseline()


if __name__ == '__main__':
    action = "pair_lm_inf"
    locals()[action]()
