import numpy as np
import scipy.special

from misc_lib import lmap, average
from tlm.estimator_prediction_viewer import EstimatorPredictionViewerGosford
from visualize.html_visual import HtmlVisualizer


def probabilty(scores, amp):
    alpha = 0.5
    seq_length = len(scores)
    scores = scores
    prob = scipy.special.softmax(scores * amp)

    p1 = np.ones_like(prob) / seq_length * alpha
    p2 = prob * (1-alpha)
    final_p = p1 + p2
    return final_p

def geo_mean_prob(scores, amp):
    alpha = 0
    scores = scores + 1
    row_sum = np.sum(scores)
    row_sum = np.expand_dims(row_sum, 0)
    prob = np.divide(scores, row_sum)
    #prob = scipy.special.softmax(scores * amp)
    final_p = prob * (1-alpha)
    return final_p

def loss_to_score(loss1, loss2):
    return loss_to_prob(loss2)/ loss_to_prob(loss1) + 0.01

def loss_to_prob(loss_l):
    return np.exp(-loss_l)



def doit(filename):
    name = filename.split(".")[0]
    bin_fn, mean_d, std_d = statistics_tlm()

    def get_score(p1, p2):
        key = bin_fn(p1)
        v = min(p2, p1)
        return (v - mean_d[key]) / std_d[key]

    st_list = []
    ed_list = []
    std_list = []
    mean_list = []
    for key in mean_d:
        st, ed = key
        st_list.append(st)
        ed_list.append(ed)
        std_list.append(std_d[key])
        mean_list.append(mean_d[key])

    mean_list = np.expand_dims(np.array(mean_list), 0)
    std_list = np.expand_dims(np.array(std_list), 0)
    st_list = np.expand_dims(np.array(st_list), 0)
    ed_list = np.expand_dims(np.array(ed_list), 0)

    def get_scores_lin(prob1_list, prob2_list):
        v2 = np.min(np.stack([prob1_list, prob2_list], axis=1), axis=1)
        v2 = np.expand_dims(v2, 1)
        all_scores = (v2 - mean_list) / std_list
        prob1_list = np.expand_dims(prob1_list, 1)
        f1 = np.less_equal(st_list, prob1_list)
        f2 = np.less(prob1_list, ed_list)
        f = np.logical_and(f1, f2)
        all_scores = all_scores * f
        scores = np.sum(all_scores, axis=1)
        return scores

    data = EstimatorPredictionViewerGosford(filename)
    amp = 0.5
    html_writer = HtmlVisualizer("{}_{}.html".format(name, amp), dark_mode=False)

    for inst_i, entry in enumerate(data):
        if inst_i > 10:
            break
        tokens = entry.get_mask_resolved_input_mask_with_input()
        scores = entry.get_vector("priority_score")
        loss1 = entry.get_vector("lm_loss1")
        loss2 = entry.get_vector("lm_loss2")
        #scores1 = get_scores_lin(loss_to_prob(loss1), loss_to_prob(loss2))
        #scores = [get_score(v1, v2) for v1,v2 in zip(loss_to_prob(loss1), loss_to_prob(loss2))]
        #assert np.all(np.less(np.abs(scores - scores1), 0.01))

        prob_scores = probabilty(scores, amp)
        prob_strs = ["{:06.6f}".format(v*1000) for v in prob_scores]

        def normalize(prob):
            # 0-> Good
            # -1 -> Bad
            return min(prob * 10000, 100)

        norm_scores = lmap(normalize, prob_scores)
        cells = data.cells_from_tokens(tokens, norm_scores)
        cells2 = data.cells_from_anything(scores, norm_scores)
        cells3 = data.cells_from_anything(prob_strs, norm_scores)
        cells4 = data.cells_from_anything(loss_to_prob(loss1), norm_scores)
        cells5 = data.cells_from_anything(loss_to_prob(loss2), norm_scores)
        html_writer.multirow_print_from_cells_list([cells, cells2, cells3, cells4, cells5])
        html_writer.write_headline("")

def show_tlm2():
    filename = "tlm_view.pickle"
    filename = "blc_cold_scores.pickle"
    filename = "blc_7show.pickle"
    doit(filename)


def get_bin_fn_from_interval(begin, end, interval):
    interval_list = []
    for st in np.arange(begin, end, interval):
        ed = st + interval
        interval_list.append((st, ed))

    def bin_fn(v):
        for st, ed in interval_list:
            if st <= v < ed:
                return st, ed
        return "Unidentifed"
    return bin_fn


def statistics_tlm():
    filename = "blc_cold_scores.pickle"
    data = EstimatorPredictionViewerGosford(filename)

    bins = {}
    bin_fn = get_bin_fn_from_interval(0, 1.05, 0.05)
    for inst_i, entry in enumerate(data):
        loss1 = entry.get_vector("lm_loss1")
        loss2 = entry.get_vector("lm_loss2")

        prob1 = loss_to_prob(loss1)
        prob2 = loss_to_prob(loss2)
        tokens = entry.get_mask_resolved_input_mask_with_input()

        for i, _ in enumerate(tokens):
            key = bin_fn(prob1[i])
            if key not in bins:
                bins[key] = []
            bins[key].append(prob2[i])

    keys = list([k for k in bins.keys() if not k == "Unidentifed"])
    keys.sort(key=lambda x:x[0])

    mean_dict = {}
    std_dict = {}
    for key in keys:
        l = average(bins[key])
        std = np.std(bins[key])
        mean_dict[key] = l
        std_dict[key] = std
        st, ed = key
        #print("{0:.2f} {1:.2f}".format(st, ed), l)
    return bin_fn, mean_dict, std_dict

def per_doc_score():
    filename = "tlm_view.pickle"
    html_writer = HtmlVisualizer("per_doc_score.html", dark_mode=False)

    data = EstimatorPredictionViewerGosford(filename)
    amp = 20
    small_threshold = 40
    for inst_i, entry in enumerate(data):
        if inst_i > 1000:
            break
        scores = entry.get_vector("priority_score")

        tokens = entry.get_mask_resolved_input_mask_with_input()
        cells = data.cells_from_tokens(tokens)
        if len(cells) < small_threshold:
            continue
        avg_score = average(scores)
        if -0.11 > avg_score > -0.30:
            continue
        print(average(scores))
        html_writer.write_headline(avg_score)
        rows = []
        row = []
        for idx, cell in enumerate(cells):
            row.append(cell)
            if len(row) == 20:
                html_writer.write_table([row])
                row = []



if __name__ == '__main__':
    show_tlm2()