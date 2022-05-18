import tensorflow as tf
from cache import save_to_pickle
import tensorflow as tf

from cache import save_to_pickle


def save_embedding_802():
    target_name = "bert/embeddings/word_embeddings"
    bert_v1 = "C:\\work\\Code\\Chair\\output\\model\\runs\\uncased_L-12_H-768_A-12\\bert_model.ckpt"
    table = tf.train.load_variable(bert_v1, target_name)
    vector = list(table[802])
    print(vector)
    save_to_pickle(vector, "bert_v1_embedding_802")
    return vector


def save_embedding_802_v2():
    target_name = "model/layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE"
    bert_v2 = "C:\\work\\Code\\Chair\\output\\model\\runs\\keras_bert\\uncased_L-12_H-768_A-12\\bert_model.ckpt"
    table = tf.train.load_variable(bert_v2, target_name)
    vector = list(table[802])
    print(vector)
    save_to_pickle(vector, "bert_v2_embedding_802")
    return vector



def main():
    v1 = save_embedding_802()
    v2 = save_embedding_802_v2()

    err = np.sum(np.abs(np.array(v1) - np.array(v2)))
    print("error {}".format(err))


if __name__ == "__main__":
    main()