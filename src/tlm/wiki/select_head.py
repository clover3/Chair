
from misc_lib import TimeEstimator

def write_stream(f, f_out):
    state = 0
    STATE_OUT_DOC = 0
    STATE_DOCNO = 1
    STATE_PRE_TEXT = 2
    STATE_IN_TEXT = 3
    STATE_POST_TEXT = 4
    content = []
    ticker = TimeEstimator(8988679, "job ", 3000)
    skip_content = False
    for line in f:
        ticker.tick()
        if not skip_content:
            content.append(line)

        if line[0] == "<":
            if state == STATE_OUT_DOC:
                assert line.startswith("<DOC>")
                state = STATE_DOCNO
            elif state == STATE_DOCNO:
                assert line.startswith("<DOCNO>")
                pre_n = len("<DOCNO>")
                ed_n = len("</DOCNO>")+1
                title = line[pre_n:-ed_n].strip()
                tokens = title.split("-")
                st, ed = tokens[-2], tokens[-1]
                if st == "0":
                    skip_content = False
                else:
                    skip_content = True
                state = STATE_PRE_TEXT
            elif state == STATE_PRE_TEXT:
                assert line.startswith("<TEXT>")
                state = STATE_IN_TEXT
            elif state == STATE_IN_TEXT:
                if line.startswith("</TEXT>"):
                    state = STATE_POST_TEXT
            elif state == STATE_POST_TEXT:
                assert line.startswith("</DOC>")
                if not skip_content:
                    for line in content:
                        f_out.write(line)
                content = []
                skip_content = False
                state = STATE_OUT_DOC
        else:
            assert state == STATE_IN_TEXT


def main():
    in_path = "/mnt/nfs/work3/youngwookim/data/tlm/enwiki_seg_galago/train.{}.trectext"
    out_path = "/mnt/nfs/work3/youngwookim/data/tlm/enwiki_heads_galago/train.{}.trectext"

    for i in range(10):
        f = open(in_path.format(i), "r")
        f_out = open(out_path.format(i), "w")
        write_stream(f, f_out)

if __name__ == "__main__":
    main()