import os

from data_generator.NLI.nli_info import corpus_dir
from data_generator.tf_gfile_support import tf_gfile

train_file = os.path.join(corpus_dir, "train.tsv")

def show():
    label_list = ["entailment", "neutral", "contradiction",]
    for idx, line in enumerate(tf_gfile(train_file, "rb")):
        if idx == 0: continue  # skip header
        line = line.strip().decode("utf-8")
        split_line = line.split("\t")
        # Works for both splits even though dev has some extra human labels.
        s1, s2 = split_line[8:10]
        l = label_list.index(split_line[-1])

        print("P:", s1)
        print("H:", s2)
        print(l)




if __name__ == "__main__":
    show()