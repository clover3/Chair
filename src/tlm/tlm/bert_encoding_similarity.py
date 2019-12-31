import os
import pickle

import numpy as np
import scipy
import tensorflow as tf

from data_generator.NLI.nli import DataLoader
from models.transformer import hyperparams
from models.transformer.tranformer_nli import transformer_nli_hidden
from path import get_bert_full_path
from path import output_path
from tlm.tlm.analyze_gradients import reshape
from trainer.model_saver import load_model_w_scope
from trainer.np_modules import get_batches_ex
from trainer.tf_train_module import init_session


def fetch_hidden_vector(hparam, vocab_size, data, model_path):
    task = transformer_nli_hidden(hparam, vocab_size, 0, False)
    sess = init_session()
    sess.run(tf.global_variables_initializer())

    load_model_w_scope(sess, model_path, ["bert"])
    batches = get_batches_ex(data, hparam.batch_size, 4)

    def batch2feed_dict(batch):
        x0, x1, x2, y  = batch
        feed_dict = {
            task.x_list[0]: x0,
            task.x_list[1]: x1,
            task.x_list[2]: x2,
            task.y: y,
        }
        return feed_dict

    def pred_fn():
        outputs = []
        for batch in batches:
            x0, x1, x2, y = batch
            all_layers, emb_outputs = sess.run([task.all_layers, task.embedding_output],
                                   feed_dict=batch2feed_dict(batch)
                                   )
            outputs.append((all_layers, emb_outputs, x0))

        return outputs

    return pred_fn()


def run():
    sequence_length = 200
    data_loader = DataLoader(sequence_length, "bert_voca.txt", True, True)
    all_data = {}
    for genre in data_loader.get_train_genres():
        items = []
        for e in data_loader.example_generator_w_genre(data_loader.train_file, genre):
            items.append(e)
            if len(items) > 50:
                break

        all_data[genre] = items

    flatten_inputs =[]
    for key, items in all_data.items():
        flatten_inputs.extend(items)

    voca_size = 30522
    hp = hyperparams.HPBert()
    model_path = get_bert_full_path()
    r = fetch_hidden_vector(hp, voca_size, flatten_inputs, model_path)
    pickle.dump(r, open(os.path.join(output_path, "hv_bert_nli.pickle"), "wb"))


def cossim(v1, v2):
    v1 = np.average(v1, axis=1)
    v2 = np.average(v2, axis=1)

    result = []
    for layer_idx in range(13):
        s = scipy.spatial.distance.cosine(v1[layer_idx], v2[layer_idx])
        result.append(s)
    return result


def max_cossim(v1, v2):
    def max_cossim_inner(v1, v2):
        max_arr = []
        for seq_idx in range(len(v1)):
            max_sim = 0
            # item1 = np.expand_dims(v1[seq_idx], 0)
#            all_item2 = v2
#            b = all_item2 / norm(all_item2)
#            cos_sim = np.dot(item1, b) / norm(item1)

            # s_list = scipy.spatial.distance.cosine(item1, all_item2)
            # max_sim = np.max(cos_sim)
            for seq_idx2 in range(len(v2)):
               s = scipy.spatial.distance.cosine(v1[seq_idx], v2[seq_idx2])
               if s > max_sim:
                   max_sim = s
            max_arr.append(max_sim)
        return np.average(max_arr)

    result = []
    for layer_idx in range(13):
        s = max_cossim_inner(v1[layer_idx], v2[layer_idx])
        result.append(s)
    return result


def analyze():
    r = pickle.load(open(os.path.join(output_path, "hv_bert_nli.pickle"), "rb"))
    r, x_list = reshape(r)
    print(len(r))

    data_loader = DataLoader(400, "bert_voca.txt", True, True)
    all_data = {}
    idx = 0
    for genre in data_loader.get_train_genres():
        print(genre)
        all_data[genre] = r[idx:idx+51]
        idx += 51

    print(r[0].shape)

    for genre, data in all_data.items():
        sim_list = []
        for _ in range(10):
            i1, i2 = np.random.choice(50, 2)
            sim = cossim(data[i1], data[i2])
            sim_list.append(sim)

        avg_sim = np.average(np.stack(sim_list, 0), axis=0)
        s = "\t".join(["{0:.2f}".format(f) for f in avg_sim])
        print(genre)
        print("In domain:\t", s)
        for genre2 in all_data.keys():
            if genre == genre2:
                continue
            sim_list = []
            for _ in range(10):
                i1, i2 = np.random.choice(50, 2)
                data2 = all_data[genre2]
                sim = cossim(data[i1], data2[i2])
                sim_list.append(sim)
            avg_sim = np.average(np.stack(sim_list, 0), axis=0)
            s = "\t".join(["{0:.2f}".format(f) for f in avg_sim])
            print("-{}: \t".format(genre2), s)


if __name__ == "__main__":
    analyze()