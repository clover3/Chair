import os
import pickle

import tensorflow as tf

from data_generator.NLI.nli import DataLoader
from models.transformer import hyperparams
from models.transformer.tranformer_nli import transformer_nli_hidden
from path import get_bert_full_path
from path import output_path
from trainer.model_saver import load_bert_v2
from trainer.np_modules import get_batches_ex
from trainer.tf_train_module_v2 import init_session


def fetch_hidden_vector(hparam, vocab_size, data, model_path):
    task = transformer_nli_hidden(hparam, vocab_size, 0, False)
    sess = init_session()
    sess.run(tf.global_variables_initializer())

    load_bert_v2(sess, model_path)
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
            all_layers, emb_outputs = sess.run([vars],
                                   feed_dict=batch2feed_dict(batch)
                                   )
            outputs.append((all_layers, emb_outputs, x0))

        return outputs

    return pred_fn()


def run():
    sequence_length = 400
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



if __name__ == "__main__":
    run()