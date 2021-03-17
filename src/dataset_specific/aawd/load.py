import glob
import os

from cpath import data_path


def parse_file(file_path):
    lines = open(file_path).readlines()
    heads = lines[0].split("\t")
    data = [line.split("\t") for line in lines[1:] if not line.startswith(';')]
    idx_pos = heads.index("positive")
    idx_neg = heads.index("negative")
    idx_text= heads.index("transcript")

    pos_list = []
    neg_list = []
    neu_list = []
    for line in data:
        if line[idx_pos] == 'true':
            pos_list.append(line[idx_text])
        elif line[idx_neg] == 'true':
            neg_list.append(line[idx_text])
        else:
            neu_list.append(line[idx_text])

    return pos_list, neg_list, neu_list



def load_all_as_text_list():
    dirPath = os.path.join(data_path, "AAWD1.1", "1.1", "english", "alignment", "merged")
    files = glob.glob(dirPath + "*.xtdf")
    pos_list = []
    neg_list = []
    neu_list = []
    for file_path in files:
        (r1, r2, r3) = parse_file(file_path)
        pos_list += r1
        neg_list += r2
        neu_list += r3
    return neg_list, neu_list, pos_list


def load_and_save():
    neg_list, neu_list, pos_list = load_all_as_text_list()

    open("agree.txt","w").write("\n".join(pos_list))
    open("neutral.txt","w").write("\n".join(neu_list))
    open("disagree.txt","w").write("\n".join(neg_list))
