import os

import numpy as np
from sklearn.metrics import mean_absolute_error

from base_type import FileName
from cpath import pjoin, output_path
from misc.show_checkpoint_vars import load_checkpoint_vars


def get_embedding_table(model_path):
    vars = load_checkpoint_vars(model_path)
    return vars['bert/embeddings/word_embeddings']


def get_nli_and_bert_embeddings():
    model_dir = pjoin(pjoin(output_path, FileName("model")), FileName("runs"))
    nli = os.path.join(model_dir, FileName("nli"), FileName("model.ckpt-75000"))
    bert = os.path.join(model_dir, FileName("uncased_L-12_H-768_A-12"), FileName("bert_model.ckpt"))
    nli_emb = get_embedding_table(nli)
    bert_emb = get_embedding_table(bert)
    return bert_emb, nli_emb


def show_embedding_difference(model_A_path, model_B_path):
    embedding_1 = get_embedding_table(model_A_path)
    embedding_2 = get_embedding_table(model_B_path)

    for i in range(30522):
        v1 = embedding_1[i]
        v2 = embedding_2[i]
        l1_dist = np.sum(np.abs(v1-v2)) / len(v1)
        l2_dist = np.sum(np.square(v1 - v2)) / len(v1)
        print(i, l1_dist, l2_dist)

    mae = mean_absolute_error(embedding_1, embedding_2)
    print("mae", mae)


def show_bert_nli_diff():
    model_dir = pjoin(pjoin(output_path, FileName("model")), FileName("runs"))
    nli = os.path.join(model_dir, FileName("nli"), FileName("model.ckpt-75000"))
    bert = os.path.join(model_dir, FileName("uncased_L-12_H-768_A-12"), FileName("bert_model.ckpt"))

    show_embedding_difference(bert, nli)


if __name__ == "__main__":
    show_bert_nli_diff()