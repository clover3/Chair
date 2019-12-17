import sys

import os
import path
from cache import DumpPickle, DumpPickleLoader
from data_generator import tokenizer_wo_tf as tokenization
from misc_lib import TimeEstimator, lmap, flatten
from tlm.wiki.sample_segments import EndofDocument


def tokenize_stream(in_file, out_path):
    dp = DumpPickle(out_path)
    vocab_file = os.path.join(path.data_path, "bert_voca.txt")
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)

    in_f = open(in_file, "r")

    def read_doc(f):
        line = f.readline()
        if not line:
            raise EndofDocument()

        assert "<DOC>" in line
        line = f.readline()
        assert "<DOCNO>" in line
        pre_n = len("<DOCNO>")
        ed_n = len("</DOCNO>") + 1
        title = line[pre_n:-ed_n].strip()

        line = f.readline()
        assert "<TEXT>" in line
        content = []
        line = f.readline()
        while line.strip() != "</TEXT>":
            content.append(line)
            line = f.readline()
        line = f.readline()
        assert "</DOC>" in line
        return title, content

    try:
        ticker = TimeEstimator(1285381, "reader", 100)
        while True:
            title, content = read_doc(in_f)
            tokens = flatten(lmap(tokenizer.tokenize,content))
            dp.dump(title, tokens)
            ticker.tick()
    except EndofDocument as e:
        pass
    dp.close()



class UnifiedReader:
    def __init__(self, saved_path_list):
        self.dp_loc = {}
        self.dpl_list = []
        for i, path in enumerate(saved_path_list):
            dpl = DumpPickleLoader(path)
            self.dpl_list.append(dpl)
            for key in dpl.loc_dict:
                self.dp_loc[key] = i

    def load(self, name):
        if name in self.dp_loc:
            dpl = self.dpl_list[self.dp_loc[name]]
            return dpl.load(name)
        else:
            raise ValueError("Key not found : " + name)

def load_seg_token_readers():
    form = '/mnt/nfs/work3/youngwookim/data/tlm_simple/seg_tokens/{}'
    path_list = list([form.format(i) for i in range(10)])
    return UnifiedReader(path_list)


def main():
    task_id = int(sys.argv[1])
    p = "/mnt/nfs/work3/youngwookim/data/tlm/enwiki_heads_galago/train.{}.trectext".format(task_id)
    out_path = '/mnt/nfs/work3/youngwookim/data/tlm_simple/seg_tokens/{}'.format(task_id)
    tokenize_stream(p, out_path)


if __name__ == "__main__":
    main()
