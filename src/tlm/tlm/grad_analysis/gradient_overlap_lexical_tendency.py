import math

from misc_lib import DictValueAverage
from tlm.estimator_prediction_viewer import EstimatorPredictionViewerGosford
from tlm.tlm.view_bfn_loss import combine


def lexical_tendency():
    filename = "nli_lm_feature_overlap\\11.pickle"
    print("Loading file")
    data = EstimatorPredictionViewerGosford(filename)
    print("Done")

    dva_list = list([DictValueAverage() for _ in range(12)])

    for inst_i, entry in enumerate(data):
        tokens = entry.get_mask_resolved_input_mask_with_input()
        h_overlap = entry.get_vector('h_overlap')  #[num_layer, seq_length, hidden_dim]
        for layer_i in range(12):
            scores = h_overlap[layer_i, :]

            for loc in range(1, 512):
                bigram = combine(tokens[loc-1], tokens[loc])
                if math.isnan(scores[loc]) or math.isinf(scores[loc]):
                    pass
                else:
                    dva_list[layer_i].add(bigram, scores[loc])

    print("Total data:", data.data_len)
    for layer_i in range(12):
        all_avg = dva_list[layer_i].all_average()

        l = list(all_avg.items())
        l.sort(key=lambda x:x[1], reverse=True)
        print("Layer : ", layer_i)
        print_cnt = 0
        print("Top-k")
        for k, v in l:
            if dva_list[layer_i].cnt_dict[k] > 10:
                print(k, v)
                print_cnt += 1

            if print_cnt > 10:
                break

        print("Low-k")
        print_cnt = 0
        for k, v in l[::-1]:
            if dva_list[layer_i].cnt_dict[k] > 100:
                print(k, v)
                print_cnt += 1

            if print_cnt > 10:
                break

if __name__ == '__main__':
    lexical_tendency()