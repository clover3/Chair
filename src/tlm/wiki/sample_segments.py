import os
from misc_lib import TimeEstimator
from cache import load_from_pickle, StreamPickler
from random import randint
from tlm.wiki.parse import parse_title_line

class EndofDocument(EOFError):
    pass

## Wikipedia has 10M segments in it.

class TextSampler:
    def __init__(self, path_list):
        self.path_list = path_list
        self.file_idx = -1
        self.cur_f = None
        self.load_next_doc()
        self.prev_seg = None
        self.cur_seg = None
        self.next_seg = None
        self.buffer = []

    def load_next_doc(self):
        self.file_idx = (self.file_idx + 1) % len(self.path_list)
        self.cur_f = open(self.path_list[self.file_idx], "r")

    def sample(self):
        try:
            self.jump()
        except EndofDocument as e:
            self.load_next_doc()
            self.jump()

        prev_segment = self.prev_seg if self.cur_seg[0] == self.prev_seg[0] else None
        cur_segment = self.cur_seg
        next_segment = self.next_seg if self.next_seg is not None and self.cur_seg[0] == self.next_seg[0] else None
        return cur_segment, prev_segment, next_segment

    def read_next_seg(self, raise_if_end):
        line = self.cur_f.readline()
        if not line:
            if raise_if_end:
                raise EndofDocument()
            else:
                return None

        assert "<DOC>" in line
        line = self.cur_f.readline()
        assert "<DOCNO>" in line
        title, st, ed = parse_title_line(line)
        line = self.cur_f.readline()
        assert "<TEXT>" in line
        content = []
        line = self.cur_f.readline()
        while line.strip() != "</TEXT>":
            content.append(line)
            line = self.cur_f.readline()
        line = self.cur_f.readline()
        assert "</DOC>" in line
        return (title, content, st, ed)

    def skip_seg(self, times):
        for i in range(times):
            line = self.cur_f.readline()
            if not line:
                raise EndofDocument()

            assert "<DOC>" in line
            line = self.cur_f.readline()
            assert "<DOCNO>" in line
            line = self.cur_f.readline()
            assert "<TEXT>" in line
            line = self.cur_f.readline()
            while line.strip() != "</TEXT>":
                line = self.cur_f.readline()
            line = self.cur_f.readline()
            assert "</DOC>" in line

    def jump(self):
        step = randint(1, 12)
        if step == 1 and self.next_seg is not None:
            self.prev_seg = self.cur_seg
            self.cur_seg = self.next_seg
            self.next_seg = self.read_next_seg(False)
        elif step == 2 and self.next_seg is not None:
            self.prev_seg = self.next_seg
            self.cur_seg = self.read_next_seg(True)
            self.next_seg = self.read_next_seg(False)
        else:
            self.skip_seg(step-1)
            self.prev_seg = self.read_next_seg(True)
            self.cur_seg = self.read_next_seg(True)
            self.next_seg = self.read_next_seg(False)
            assert self.cur_seg is not None



def main():
    num_inst = 1000 * 1000 * 100
    path_format = "/mnt/nfs/work3/youngwookim/data/tlm/enwiki_seg_galago/train.{}.trectext"
    text_path = list([path_format.format(i) for i in range(10)])

    ts = TextSampler(text_path)
    sp = StreamPickler("wiki_segments3_", 1000*100)
    ticker = TimeEstimator(num_inst)
    for i in range(num_inst):
        inst = ts.sample()
        sp.add(inst)
        ticker.tick()

    sp.flush()


if __name__ == "__main__":
    main()
