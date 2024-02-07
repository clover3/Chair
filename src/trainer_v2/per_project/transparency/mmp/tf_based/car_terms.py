import pickle
from krovetzstemmer import Stemmer
from cpath import output_path
from misc_lib import path_join
import math

from tab_print import print_table

if __name__ == '__main__':
    tf_path = path_join(output_path, "mmp", "lucene_krovetz", "tf.pkl")
    bg_tf = pickle.load(open(tf_path, "rb"))
    bg_sum = sum(bg_tf.values())

    stemmer = Stemmer()
    tf_bar_car = pickle.load(open("output/mmp/car_rel_tf/all.pkl", "rb"))
    tf_bar_car_stem = {}
    for term, v in tf_bar_car.items():
        term_stemmed = stemmer.stem(term)
        if term_stemmed not in tf_bar_car_stem:
            tf_bar_car_stem[term_stemmed] = 0
        tf_bar_car_stem[term_stemmed] += v


    rel_sum = sum(tf_bar_car_stem.values())
    not_found = []

    def log_odd(rel_tf, bg_tf):
        return math.log(rel_tf / rel_sum) - math.log(bg_tf / bg_sum)

    triplets = []
    for term, tf in tf_bar_car_stem.items():
        bg_val = bg_tf[term]
        if bg_val == 0:
            not_found.append(term)
        else:
            triplets.append((term, tf, bg_val))

    print(len(not_found), "terms not found")
    table = []
    for k, v1, v2 in triplets:
        if v1 > 5:
            row = [k, v1, v2, log_odd(v1, v2) * v1]
            table.append(row)
    table.sort(key=lambda x: x[3], reverse=True)
    print_table(table)


