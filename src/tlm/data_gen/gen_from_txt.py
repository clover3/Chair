
from data_generator.common import get_tokenizer
from tlm.data_gen.base import UnmaskedPairGen


class GenFromTxt(UnmaskedPairGen):
    def __init__(self):
        super(GenFromTxt, self).__init__()
        self.tokenizer = get_tokenizer()

    def load_doc(self, text_file):
        lines = open(text_file, "r").readlines()

        segments = []
        for l in lines:
            segments.append(self.tokenizer.tokenize(l))

        return segments
