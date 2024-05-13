import tensorflow as tf
from cache import load_pickle_from
from cpath import output_path
from misc_lib import path_join
from math import isclose

def main():
    tf_pickle_path = path_join(output_path, "mmp", "lucene_krovetz", "tf.pkl")
    tf_d = load_pickle_from(tf_pickle_path)
    ctf = sum(tf_d.values())

    bg_prob_pickle_path = path_join(output_path, "mmp", "lucene_krovetz", "bg_prob.pickle")
    bg_prob_d = load_pickle_from(bg_prob_pickle_path)


    for term, tf in tf_d.items():
        bg_prob1 = tf / ctf
        bg_prob2 = bg_prob_d[term]

        if not isclose(bg_prob1, bg_prob2):
            print("{}\t{}\t{}".format(term, bg_prob1, bg_prob2))

    print("Checked {} terms.".format(len(tf_d)))
    print(len(bg_prob_d) == len(tf_d))

if __name__ == "__main__":
    main()
