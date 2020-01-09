from collections import Counter

import numpy as np

import cpath
from cache import *
from models.transformer import bert
from models.transformer import hyperparams
from models.transformer.bert import *


def fetch_bert_parameter(model_path):
    hp = hyperparams.HPSENLI()
    vocab_size = 30522
    vocab_filename = "bert_voca.txt"

    config = bert.BertConfig(vocab_size=vocab_size,
                             hidden_size=hp.hidden_units,
                             num_hidden_layers=hp.num_blocks,
                             num_attention_heads=hp.num_heads,
                             intermediate_size=hp.intermediate_size,
                             type_vocab_size=hp.type_vocab_size,
                             )

    hp.compare_deletion_num = 20
    seq_length = hp.seq_max

    is_training = False
    input_ids = tf.placeholder(tf.int64, [None, seq_length])
    input_mask = tf.placeholder(tf.int64, [None, seq_length])
    segment_ids = tf.placeholder(tf.int64, [None, seq_length])
    label_ids = tf.placeholder(tf.int64, [None])
    use_one_hot_embeddings = False
    model = bert.BertModel(
        config=config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False
                            )
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    variables = tf.contrib.slim.get_variables_to_restore()
    for v in variables:
        print(v)

    names = list([v.name for v in variables])
    loader = tf.train.Saver()
    loader.restore(sess, model_path)
    r, = sess.run([variables])

    output = dict(zip(names, r))

    for k in output:
        print(k)

    return output


def get_path_from_model_id(load_id):
    run_dir = os.path.join(cpath.output_path, "model", 'runs')
    save_dir = os.path.join(run_dir, load_id[0])
    return os.path.join(save_dir, "{}".format(load_id[1]))


def load_parameter():
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    bert_params = fetch_bert_parameter(get_path_from_model_id(load_id))
    tf.reset_default_graph()
    load_id = ("NLI_run_A", 'model-0')
    nli_params = fetch_bert_parameter(get_path_from_model_id(load_id))

    save_to_pickle(bert_params, "bert_params")
    save_to_pickle(nli_params, "nli_params")


def is_near_zero(v):
    return abs(v) < 1e-3

def count_zero(l):
    l = np.abs(l)
    l = l < 1e-2
    return np.count_nonzero(l)


def count_less_than_one(l):
    l = np.abs(l)
    l = l < 1
    return np.count_nonzero(l)


def analyze_parameter():
    bert_params = load_from_pickle("bert_params")
    nli_params = load_from_pickle("nli_params")

    def square_dist_sum(m1, m2):
        return np.sum((m1 - m2)**2)

    def square_dist_avg(m1, m2):
        return np.average((m1 - m2)**2)

    def dist_sum(m1, m2):
        return np.sum(np.abs(m1 - m2))

    def dist_max(m1, m2):
        return np.max(np.abs(m1 - m2))

    def dist_avg(m1, m2):
        return np.average(np.abs(m1 - m2))

    def num_elems(shape):
        v = 1
        for l in shape:
            v = v*l
        return v
    layer_sum = Counter()
    acc_bert = 1
    acc_nli = 1
    for key in bert_params:
        v1 = bert_params[key]
        v2 = nli_params[key]

        sum_d = dist_sum(v1, v2)
        avg_d = dist_avg(v1, v2)
        max_d = dist_max(v1, v2)
        #print("{}\t{}\t{}\t{}\t{}".format(key, v1.shape, sum_d, avg_d, max_d))



        n = num_elems(v1.shape)
        n_zero_bert = count_zero(v1)
        n_zero_nli = count_zero(v2)
        if "bias" not in key and "attention" not in key:
            acc_bert *= (n-n_zero_bert)
            acc_nli *= (n - n_zero_nli)
            if "LayerNorm" not in key:
                print("{}\t{}\t{}\t{}\t{}".format(key, v1.shape, n_zero_bert, n_zero_nli, n_zero_bert-n_zero_nli))

        tokens = key.split("/")
        if tokens[2].startswith("layer_"):
            layer_sum[tokens[2]] += sum_d

        if tokens[1].startswith("embeddings"):
            layer_sum[tokens[1]] += sum_d

    print("BERT/NLI", acc_bert/acc_nli)
    for key in layer_sum:
        print(key, layer_sum[key])

if __name__ == '__main__':
    action = "analyze_parameter"
    locals()[action]()

