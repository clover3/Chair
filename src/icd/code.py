import pickle
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras import Input, Model

from icd.common import lmap, load_description, AP_from_binary
from misc_lib import flatten


def encode_data(vocabulary_set, text_list):
    encoder = tf.keras.preprocessing.text.text_to_word_sequence(vocabulary_set)
    return lmap(encoder.encode, text_list)


str_code_id = 'input_A'
str_desc_tokens = 'desc_tokens'


def input_fn(data, tokenizer, max_seq):
    input2 = lmap(lambda x: x['short_desc'], data)
    ids = lmap(lambda x:x['order_number'], data)

    code_ids = list([[e] for e in ids])

    input2 = lmap(tf.keras.preprocessing.text.text_to_word_sequence, input2)
    enc_text = tokenizer.texts_to_sequences_generator(input2)

    def fit_to_length(e):
        e = e[:max_seq]
        e = e + (max_seq - len(e)) * [0]
        return [e]

    enc_desc_tokens = lmap(fit_to_length, enc_text)
    dataset = tf.data.Dataset.from_tensor_slices({str_code_id: code_ids, str_desc_tokens:enc_desc_tokens})
    dataset = dataset.shuffle(1000)
    dataset = dataset.repeat()
    return dataset


class MyLayer(tf.keras.layers.Layer):
    def __init__(self, n_output_voca, dim, max_seq, **kwargs):
        self.n_output_voca = n_output_voca
        self.dim = dim
        self.max_seq  = max_seq
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        #self.W = tf.Variable(tf.random_normal([self.dim, self.n_output_voca]))
        self.W = self.add_weight(name='kernel',
                                      shape=(self.dim, self.n_output_voca),
                                      initializer='uniform',
                                      trainable=True)

        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end


    def call(self, x):
        x = tf.reshape(x, [-1, self.dim])
        logits = tf.matmul(x, self.W)
        #logits = backend.dot(x, self.W)
        logits = tf.reshape(logits, [-1, self.max_seq, self.n_output_voca])
        return logits


def build_model(dim, max_seq, n_input_voca, n_output_voca):
    #W2 = tf.keras.layers.Dense(n_output_voca, use_bias=False)
    #W2 = K.random_normal_variable(shape=(n_output_voca, dim), mean=0, scale=1)
    np_val = np.reshape(np.random.normal(size=n_output_voca * dim), [n_output_voca, dim])
    W2 = K.constant(np_val)

    code_id = Input(shape=[1], name=str_code_id)
    token_ids = Input(shape=(max_seq,), dtype=tf.int32, name=str_desc_tokens)

    W1 = tf.keras.layers.Embedding(n_input_voca, dim)
    h0 = W1(code_id)

    #logits = W2(h)
    #logits = MyLayer(n_output_voca, dim, max_seq)(h)
    h = tf.reshape(h0, [-1, dim])
    h = tf.nn.l2_normalize(h, -1)
    W2 = tf.nn.l2_normalize(W2, -1)
    logits = tf.matmul(h, W2, transpose_b=True)
    #logits = tf.reshape(logits, [-1, max_seq, n_output_voca])

    log_probs = tf.nn.log_softmax(logits, axis=-1)

    y = tf.one_hot(token_ids, depth=n_output_voca) #[batch, max_seq, n_output_voca]

    pos_val = logits * y # [ batch, max_seq, voca]
    neg_val = logits - tf.reduce_sum(pos_val, axis=1) #[ batch, voca]
    t = tf.reduce_sum(pos_val, axis=2) # [batch, max_seq]
    correct_map = tf.expand_dims(t, 2) # [batch, max_seq, 1]
    print(correct_map.shape)
    #logits_correct = tf.expand_dims(tf.reduce_sum(logits * y, axis=-1), -1)
    wrong_map = tf.expand_dims(neg_val, 1) # [batch, 1, voca]
    print(wrong_map.shape)
    t = wrong_map - correct_map + 1
    print(t.shape)
    loss = tf.reduce_mean(tf.math.maximum(t, 0), axis=-1)
    mask = tf.cast(tf.not_equal(token_ids, 0), tf.float32) # batch, seq_len
    print(mask.shape)
#    loss = -tf.reduce_sum(input_tensor=log_probs * y, axis=[-1])
    loss = mask * loss
    loss = tf.reduce_sum(loss, axis=1) # over the sequence
    loss = tf.reduce_mean(loss)
    print(loss.shape)
    model = Model(inputs=[code_id, token_ids], outputs=h0)
    return loss, model


def fit_and_tokenize(input2):
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(input2)
    input2 = lmap(tf.keras.preprocessing.text.text_to_word_sequence, input2)
    print(input2[0])
    enc_text = tokenizer.texts_to_sequences_generator(input2)
    return enc_text, tokenizer



def work(args):
    data = load_description()
    n_input_voca = data[-1]['order_number']
    input2 = lmap(lambda x: x['short_desc'], data)
    dim = 1000
    max_seq = 1
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    lr = float(args.lr)
    print("learning rate", lr)
    print("Batch_size", batch_size)
    enc_text, tokenizer = fit_and_tokenize(input2)

    random.shuffle(data)
    train_size = int(0.1 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]

    step_per_epoch = int(len(train_data) / batch_size)

    train_dataset = input_fn(train_data, tokenizer, max_seq)
    val_dataset = input_fn(val_data, tokenizer, max_seq)
    token_config = tokenizer.get_config()
    n_output_voca = len(token_config['word_index'])
    loss, model = build_model(dim, max_seq, n_input_voca, n_output_voca)
    model.add_loss(loss)
    #model = multi_gpu_model(model, 4, cpu_relocation=True)
    optimizer = tf.keras.optimizers.Adam(
        lr=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(optimizer=optimizer)
    model.fit(train_dataset,
             # validation_data=val_dataset,
             # validation_steps=3000,
              epochs=epochs, steps_per_epoch=step_per_epoch, batch_size=batch_size)
    model.save('my_model.h5')


def work2():
    data = load_description()
    n_input_voca = data[-1]['order_number']
    input2 = lmap(lambda x: x['short_desc'], data)
    dim = 100
    max_seq = 30

    all_text = tokenize(input2)
    voca = set(flatten(all_text))

    word2idx = {}
    for idx, word in enumerate(list(voca)):
        word2idx[word] = idx

    def tokens_to_idx(tokens):
        return list([word2idx[t] for t in tokens])

    random.shuffle(data)
    train_size = int(0.9 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    train_ids = lmap(lambda x:x['order_number'], train_data)
    icd10_codes = lmap(lambda x: x['icd10_code'], train_data)
    train_desc = lmap(lambda x: x['short_desc'], train_data)
    train_tokens = lmap(tokens_to_idx, train_desc)

    n_output_voca = len(voca)
    print("n_output_voca", n_output_voca)
    W2 = np.random.rand(n_output_voca+1, dim)
    W2 = np.random.normal(0, 1, [n_output_voca+1, dim])
    W1 = np.zeros([n_input_voca+1, dim])

    icd10_codes = lmap(lambda x: x.strip(), icd10_codes)

    add_subword = False

    code_id_to_code = {}
    code_to_code_id = {}
    for code_id, icd10_code, text_seq in zip(train_ids, icd10_codes, train_tokens):
        for idx in text_seq:
            W1[code_id] += W2[idx]

        code_id_to_code[code_id] = icd10_code
        code_to_code_id[icd10_code] = code_id

        l = len(icd10_code)
        if add_subword:
            for j in range(1, l-1):
                substr = icd10_code[:j]
                if substr in code_id_to_code:
                    W1[code_id] += W1[code_to_code_id[substr]]

    new_w2 = list([W2[i] / np.linalg.norm(W2[i]) for i in range(n_output_voca)])

    all_voca = []
    for code_id in train_ids:
        icd10_code = code_id_to_code[code_id]
        word = icd10_code
        emb = W1[code_id]
        all_voca.append((word, emb))

    print("Testing")
    AP_list = []
    for code_id, text_seq in zip(train_ids, train_tokens):
        a = W1[code_id] / np.linalg.norm(W1[code_id])
        l = []
        for j in range(n_output_voca):
            b = new_w2[j]
            e = j, np.dot(a, b)
            l.append(e)

        l.sort(key=lambda x:x[1], reverse=True)
        ranked_list = l[:50]
        terms = text_seq
        def is_correct(w_pair):
            return w_pair[0] in terms
        AP = AP_from_binary(lmap(is_correct, ranked_list), len(terms))
        AP_list.append(AP)
        print("1")
        if len(AP_list) > 100:
            break

    print(sum(AP_list)/len(AP_list))

def save_analysis():
    obj = pickle.load(open("sent.w2v", "rb"))

if __name__ == "__main__":
    work2()
    # args = parser.parse_args(sys.argv[1:])
    # work(args)