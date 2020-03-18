import os

from list_lib import lmap

run_id = 3
os.environ["CUDA_VISIBLE_DEVICES"] = str(run_id)

import tensorflow as tf
from misc_lib import *
from models.PSF import get_relevant_docs
from nltk import sent_tokenize
import pickle
from arg.predictor import Predictor
from data_generator.argmining.NextSentPred import get_pseudo_label_path


def pred_stance(target_topic):
    docs = get_relevant_docs(target_topic)
    sents_list = lmap(sent_tokenize, docs)

    predictor = Predictor(target_topic)
    def get_topic_stance(sents, target_topic):
        return predictor.predict(target_topic, sents)

    topic_stances_list = flat_apply_stack(lambda x: get_topic_stance(x, target_topic), sents_list, False)
    save_path = get_pseudo_label_path(target_topic)
    pickle.dump(topic_stances_list, open(save_path, "wb"))



if __name__ == "__main__":
    all_topics = ["abortion", "cloning", "death_penalty", "gun_control",
                  "marijuana_legalization", "minimum_wage", "nuclear_energy"]

    st = run_id * 2
    ed = run_id * 2 + 2

    for topic in all_topics[st:ed]:
        print("-------------------")
        print(topic)
        tf.reset_default_graph()
        pred_stance(topic)
