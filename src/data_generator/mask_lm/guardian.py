from data_generator.mask_lm.chunk_lm import Text2Case
from data_generator.data_parser import guardian


class GuardianLoader(Text2Case):
    def __init__(self, topic, seq_length, shared_setting):
        text_list = guardian.load_as_text_chunk(topic)
        super(GuardianLoader, self).__init__(text_list, seq_length, shared_setting)
