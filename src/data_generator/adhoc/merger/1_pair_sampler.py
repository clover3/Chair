import random
import pickle

## read data

def read_list(i):
    st = i * 10000
    ed = (i+1) * 10000
    f = open("{}_{}.list".format(st, ed), "r")
    for line in f:
        q_id, _, doc_id, rank, score, _ = line.split()
        yield q_id, doc_id, score


## debais sampling
def sample_debias(ranked_list):
    max_occurence = 10000
    output = {}
    for q_id, doc_id, score in ranked_list:
        score_grouper = int(float(score) + 0.8)
        if score_grouper not in output:
            output[score_grouper] = []
        if len(output[score_grouper]) < max_occurence:
            output[score_grouper].append((q_id, doc_id, score))
    return output


def pick1(l):
    return l[random.randrange(len(l))]


def generate_data():
    print("reading data")
    big_i = 0
    for big_i in range(1,10):
        l = read_list(big_i)
        groups = sample_debias(l)

        def sample_one(keys, d):
            key = pick1(keys)
            entry = pick1(d[key])
            return entry

        keys = list(groups.keys())
        data_size = 10 * 1000

        print("Sampling")
        for j in range(10):
            result = []
            for i in range(data_size):
                q_id_1, doc_id_1, score1 = sample_one(keys, groups)
                q_id_2, doc_id_2, score2 = sample_one(keys, groups)

                if score1 < score2 :
                    result.append((q_id_1, doc_id_1, q_id_2, doc_id_2))
                else:
                    result.append((q_id_2, doc_id_2, q_id_1, doc_id_1))
            gid = big_i * 10 + j
            pickle.dump(result, open("merger_idpair_{}.pickle".format(gid), "wb"))


if __name__ == "__main__":
    generate_data()
