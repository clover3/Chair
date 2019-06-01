import tensorflow as tf
import os
from path import data_path


label_list = ["entailment", "neutral", "contradiction",]
corpus_dir = os.path.join(data_path, "nli")


def rewrite(filepath, out_path):
    out_f = open(out_path, "w")
    for idx, line in enumerate(tf.gfile.Open(filepath, "rb")):
        if idx == 0: continue  # skip header
        line = line.strip().decode("utf-8")
        split_line = line.split("\t")
        # Works for both splits even though dev has some extra human labels.
        s1, s2 = split_line[8:10]
        l = label_list.index(split_line[-1])

        x_str = s1 + " <SEP> " + s2
        y = [0, 0, 0]
        y[l] = 1
        y_str = " ".join([str(v) for v in y])
        sep  = "sep"

        out_f.write("{}\t{}\t{}\n".format(y_str, sep, x_str))
    out_f.close()



if __name__ == "__main__":
    train_file = os.path.join(corpus_dir, "train.tsv")
    dev_file = os.path.join(corpus_dir, "dev_matched.tsv")

    out_path = os.path.join(data_path, "train.tsv")
    rewrite(train_file, out_path)
