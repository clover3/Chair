from trainer.tf_module import *
from .baselines import get_real_len, informative_fn_eq1
from .deepexplain.tensorflow import methods

def explain_by_gradient(data, method_name, label_type, sess, de, feed_end_input, emb_outputs, emb_input, softmax_output):
    batch_size = 16
    x_input = emb_input
    encoded_embedding_in, attention_mask_in = emb_input
    stop = [encoded_embedding_in]

    def get_emb(batch):
        emb_vals = sess.run(emb_outputs, feed_dict=feed_end_input(batch))
        return emb_vals

    actual_len = []
    for entry in data:
        x0, x1, x2 = entry
        seq_len = len(x0)
        real_len = get_real_len(x1, seq_len)
        actual_len.append(real_len)
    seq_len = len(data[0][0])


    T_attrib_list = []
    for i in range(3):
        T_attrib = de.prepare(method_name, softmax_output[:, i], stop, x_input, None, None)
        T_attrib_list.append(T_attrib)

    def fetch_salience(batch):
        xi = get_emb(batch)
        stop_val = xi

        emb2logit_attribution =[]
        print(methods.total_runs)
        for i in range(3):
            fl = de.explain_prepared(T_attrib_list[i], method_name, softmax_output[:, i], stop, x_input, xi, stop_val)
            #fl = de.explain(method_name, softmax_output[:, i], stop, x_input, xi, stop_val)
            # len(fl) == 1
            # fl has shape [-1, max_seq, emb_dim]
            emb2logit_attribution.append(fl[0])
        return np.stack(emb2logit_attribution, axis=1)

    batches = get_batches_ex(data, batch_size, 3)

    emb2logit_list = list([fetch_salience(b) for b in batches])
    # list [ x.shape = [-1, 3, max_seq, emb_dim] ]

    emb2logit = np.concatenate(emb2logit_list, axis=0)
    print("Average runs:", methods.total_runs/len(emb2logit))
    assert len(emb2logit.shape) == 4
    assert emb2logit.shape[0] == len(data)
    attrib = informative_fn_eq1(label_type, emb2logit)

    num_case = len(data)
    explains = []
    for i in range(num_case):
        attrib_scores = np.zeros([seq_len])
        for idx in range(actual_len[i]):
            salience = attrib[i, idx]
            score = np.sum(salience)
            attrib_scores[idx] = score

        explains.append(attrib_scores)
    return explains

