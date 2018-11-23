

class Enwiki2Stance:
    vocab_filename = "shared_voca.txt"
    vocab_size = 32000


class Guardian2Stance:
    vocab_filename = "guardian_voca.txt"
    vocab_size = 32000


class Tweets2Stance:
    vocab_filename = "tweets_voca.txt"
    vocab_size = 32000
    seq_length = 50


class TopicTweets2Stance:
    vocab_filename = None
    vocab_size = 32000
    seq_length = 50

    def __init__(self, topic):
        self.vocab_filename = "tweets_{}_voca.txt".format(topic)
