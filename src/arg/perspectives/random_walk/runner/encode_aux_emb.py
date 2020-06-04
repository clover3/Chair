import os
import pickle

from arg.perspectives.random_walk.tfrecord_gen import gen_with_aux_emb
from cpath import data_path
from data_generator.job_runner import sydney_working_dir
from misc_lib import exist_or_mkdir


def make_dev():
    save_dir_path = os.path.join(data_path, "perspective_with_aux")
    exist_or_mkdir(save_dir_path)
    root_path = os.path.join(sydney_working_dir, "pc_dev_word2vec")
    aux_embedding_d = {}
    for i in range(112):
        key, model = pickle.load(open(os.path.join(root_path, str(i)), "rb"))
        aux_embedding_d[key] = model

    print("Working on dev")
    gen_with_aux_emb(os.path.join(save_dir_path, "dev"),
                     aux_embedding_d,
                     "dev",
                     100)


def make_train():
    save_dir_path = os.path.join(data_path, "perspective_with_aux")
    exist_or_mkdir(save_dir_path)
    root_path = os.path.join(sydney_working_dir, "pc_train_word2vec")
    aux_embedding_d = {}
    for i in range(453):
        key, model = pickle.load(open(os.path.join(root_path, str(i)), "rb"))
        aux_embedding_d[key] = model

    print("Working on train")
    gen_with_aux_emb(os.path.join(save_dir_path, "train"),
                     aux_embedding_d,
                     "train",
                     100)


if __name__ == "__main__":
    make_train()
