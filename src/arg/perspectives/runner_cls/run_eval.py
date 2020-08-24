from arg.perspectives.bm25_predict import get_classifier, tune_kernel_save
from arg.perspectives.classification import eval_classification, load_payload, get_scores
from arg.perspectives.load import get_claim_perspective_id_dict
from cache import load_from_pickle
from list_lib import flatten
from misc_lib import print_dict_tab


def main():
    param = {
        'verbose': True
    }
    cls = get_classifier(param)

    def all_true(a, b):
        return 1
    def all_false(a, b):
        return 0

    print("all_true")
    r = eval_classification(all_true, "dev")
    print_dict_tab(r)
    print("all_false")
    r = eval_classification(all_false, "dev")
    print_dict_tab(r)


def tune_kernel_a():
    split = "train"
    payloads = load_payload(split)
    gold = get_claim_perspective_id_dict()

    r = []
    for cid, data_list in payloads:
        gold_pids = gold[cid]
        all_pid_set = set(flatten(gold_pids))
        for p_entry in data_list:
            c_text = p_entry['claim_text']
            p_text = p_entry['perspective_text']
            y = 1 if p_entry['pid'] in all_pid_set else 0
            r.append((c_text, p_text, y))
    tune_kernel_save(r)


def tune_kernel_b():
    def linear_kernel(norm_max, val, k):
        if val > k * norm_max:
            return 1
        else:
            return 0

    data = load_from_pickle("bm25_tune_kernel")


    scores = []
    k = 0.4

    while k < 0.7:
        r = list([(linear_kernel(score_ideal, cur_score, k), y) for score_ideal, cur_score, y in data])
        score = get_scores(r)
        score['k'] = k
        scores.append(score)
        k += 0.01

    scores.sort(key=lambda d: d['accuracy'], reverse=True)

    print(scores[0])

if __name__ == "__main__":
    #tune_kernel_a()
    main()