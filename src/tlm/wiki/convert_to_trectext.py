
import sys
from typing import List

from misc_lib import TimeEstimator
from tlm.fast_tokenizer import FTokenizer
from tlm.wiki.split_wikipedia import WikiSpliter


def empty_line(line):
    return len(line.strip()) == 0


class Segment:
    def __init__(self, outpath):
        self.ws = WikiSpliter()
        self.ftokenizer = FTokenizer()
        self.unknown_idx = 0

        self.seg_max_seq = 256 - 3
        self.skip = int(self.seg_max_seq/2)
        self.out_f = open(outpath, "w")

    def pop(self, title, content_list: List[str]):
        content = " ".join(content_list)
        self.save(title, content)

    def get_unknown_name(self):
        self.unknown_idx += 1
        return "unknown{}_".format(self.unknown_idx)

    def save(self, title, content: str):
        def write_title(t):
            self.out_f.write("<DOCNO> {} </DOCNO>\n".format(t))

        def write_content(t):
            s = "<TEXT>\n" + t + "\n</TEXT>"
            self.out_f.write(s)

        self.out_f.write("<DOC>\n")
        write_title(title)
        write_content(content)
        self.out_f.write("\n</DOC>\n")

    def parse(self, f):
        f_prev_empty = True
        title = None
        content: List[str] = []
        ticker = TimeEstimator(12099257, "parse", 10000)
        for line in f:
            if empty_line(line):
                f_prev_empty = True
            else:
                if f_prev_empty:
                    if self.ws.is_title(line):
                        if title is not None:
                            self.pop(title, content)
                        title = line.strip()
                        content = []
                    elif title is None:
                        title = self.get_unknown_name()

                content.append(line)
                f_prev_empty = False

            ticker.tick()
        self.pop(title, content)


def parse():
    data_no = sys.argv[1]
    input_path = "/mnt/nfs/work3/youngwookim/data/enwiki4bert/enwiki_train.txt.line.{}".format(data_no)
    outpath = "/mnt/nfs/work3/youngwookim/data/wikipedia_trectext/enwiki_train.txt.{}.trectext".format(data_no)
    s = Segment(outpath)
    f = open(input_path, "r")
    s.parse(f)


if __name__ == "__main__":
    parse()
