from data_generator.mask_lm.chunk_lm import Text2Case
from data_generator.data_parser import tweets


class TweetLoader(Text2Case):
    def __init__(self, topic, seq_length, shared_setting):
        text_list = tweets.load_as_text_chunk(topic)
        super(TweetLoader, self).__init__(text_list, seq_length, shared_setting)