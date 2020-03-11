import nltk
import numpy as np
from gensim.models import KeyedVectors

from icd.common import lmap, load_description
from misc_lib import flatten


def tokenize(input2):
    return list([nltk.word_tokenize(line) for line in input2])


def save_word2vec_format(word_emb_pair_list, out_path):
    dim = len(word_emb_pair_list[0][1])
    n_word = len(word_emb_pair_list)
    f = open(out_path, "w")
    f.write("{} {}\n".format(n_word, dim))
    for word, emb in word_emb_pair_list:
        tokens = [word] + list([str(t) for t in emb])
        f.write(" ".join(tokens) + "\n")
    f.close()


def load_test():
    kv = KeyedVectors.load_word2vec_format("manual_voca.txt", binary=False)
    for key in kv.vocab:
        print(key)



def build_embedding(dim, icd10_codes, ids, n_input_voca, n_output_voca, train_tokens):
    W2 = np.random.normal(0, 1, [n_output_voca + 1, dim])
    W1 = np.zeros([n_input_voca + 1, dim])
    add_subword = True
    code_id_to_code = {}
    code_to_code_id = {}
    for code_id, icd10_code, text_seq in zip(ids, icd10_codes, train_tokens):
        for idx in text_seq:
            W1[code_id] += W2[idx]

        code_id_to_code[code_id] = icd10_code
        code_to_code_id[icd10_code] = code_id

        l = len(icd10_code)
        if add_subword:
            for j in range(1, l - 1):
                substr = icd10_code[:j]
                if substr in code_id_to_code:
                    W1[code_id] += W1[code_to_code_id[substr]]
    return W1, W2, code_id_to_code


def export_embeddings(W1, W2, code_id_to_code, ids, word2idx):
    all_voca = []
    for code_id in ids:
        icd10_code = code_id_to_code[code_id]
        word = icd10_code
        emb = W1[code_id]
        all_voca.append((word, emb))
    for word, idx in word2idx.items():
        emb = W2[idx]
        all_voca.append((word, emb))
    return all_voca


def build_voca(data):
    short_desc_list = lmap(lambda x: x['short_desc'], data)
    all_text = tokenize(short_desc_list)
    voca = set(flatten(all_text))
    n_output_voca = len(voca)
    word2idx = {}
    for idx, word in enumerate(list(voca)):
        word2idx[word] = idx
    return n_output_voca, word2idx


def extract_data(data, word2idx):
    def tokens_to_idx(tokens):
        return list([word2idx[t] for t in tokens])

    ids = lmap(lambda x: x['order_number'], data)
    icd10_codes = lmap(lambda x: x['icd10_code'], data)
    desc_list = lmap(lambda x: x['short_desc'], data)
    train_tokens = lmap(tokens_to_idx, tokenize(desc_list))
    icd10_codes = lmap(lambda x: x.strip(), icd10_codes)

    return icd10_codes, ids, train_tokens


def manually_build_embedding():
    data = load_description()
    n_input_voca = data[-1]['order_number']
    dim = 100
    n_output_voca, word2idx = build_voca(data)
    icd10_codes, ids, train_tokens = extract_data(data, word2idx)
    print("n_output_voca", n_output_voca)
    W1, W2, code_id_to_code = build_embedding(dim, icd10_codes, ids, n_input_voca, n_output_voca, train_tokens)
    all_voca = export_embeddings(W1, W2, code_id_to_code, ids, word2idx)
    save_word2vec_format(all_voca, "manual_voca.txt")


if __name__ == "__main__":
    manually_build_embedding()
