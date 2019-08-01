from tlm.split_wikipedia import WikiSpliter
from tlm.fast_tokenizer import FTokenizer
from misc_lib import TimeEstimator
import sys


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

    def pop_old(self, title, content):
        content_str = "".join(content)
        # content's first line is title

        seg_st = []
        seg_ed = []
        sub_loc = 0

        num_subword_high_estimate = int(len(content_str) / 2)
        while sub_loc < num_subword_high_estimate:
            seg_st.append(sub_loc)
            seg_ed.append(sub_loc+self.seg_max_seq)
            sub_loc += self.skip

        all_sub_loc = seg_st + seg_ed
        all_sub_loc.sort()

        corr_loc = self.ftokenizer.get_subtoken_loc(content_str, all_sub_loc)
        loc_d = dict(zip(all_sub_loc, corr_loc))


        for i in range(len(seg_st)):
            st = seg_st[i]
            ed = seg_ed[i]

            if ed not in loc_d:
                break

            st_loc = loc_d[st]
            ed_loc = loc_d[ed]
            segment = content_str[st_loc:ed_loc]
            self.save(title, st_loc, ed_loc, segment)

    def pop(self, title, content):
        content_str = "".join(content)

        segments = self.ftokenizer.smart_cut(title, content_str, self.seg_max_seq)

        for segment, st, ed in segments:
            self.save(title, st, ed, segment)


    def save(self, title, st_loc, ed_loc, segment):
        new_title = title + "-{}-{}".format(st_loc, ed_loc)
        def write_title(t):
            self.out_f.write("<DOCNO> {} </DOCNO>\n".format(t))

        def write_content(t):
            s = "<TEXT>\n" + t + "\n</TEXT>"
            self.out_f.write(s)

        self.out_f.write("<DOC>\n")
        write_title(new_title)
        write_content(segment)
        self.out_f.write("\n</DOC>\n")

    def get_unknown_name(self):
        self.unknown_idx += 1
        return "unknown{}_".format(self.unknown_idx)

    def parse(self, f):
        f_prev_empty = True
        title = None
        content = []
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
    outpath = "/mnt/nfs/work3/youngwookim/data/tlm/enwiki_seg_galago/enwiki_train.txt.line.{}".format(data_no)
    s = Segment(outpath)
    f = open("/mnt/nfs/work3/youngwookim/data/enwiki4bert/enwiki_train.txt.line.{}".format(data_no), "r")
    s.parse(f)







if __name__ == "__main__":
    print("wiki_segment")
    parse()
