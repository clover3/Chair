from collections import Counter

from contradiction.medical_claims.annotation_1.load_data import get_pair_dict
from contradiction.medical_claims.annotation_1.process_annotation import load_annots_w_processing
from list_lib import left
from misc_lib import group_by, get_first, average
from stats.agreement import cohens_kappa


def print_annot_stats(all_annots):
    key_counter = Counter(left(all_annots))
    cnt_counter = Counter()
    for hit_id, cnt in key_counter.items():
        cnt_counter[cnt] += 1
    for key, value in cnt_counter.items():
        print(key, value)


def print_annot_stats_grouped(all_annots):

    key_counter = Counter(left(all_annots))
    cnt_counter_list = [Counter() for _ in range(25)]
    for hit_id, cnt in key_counter.items():
        group_no, _ = hit_id
        cnt_counter_list[group_no][cnt] += 1

    for group_no in range(1, 25):
        print("Group ", group_no)
        for key, value in cnt_counter_list[group_no].items():
            print(key, value)


def do_agreement_analysis(pair_d, all_annots):
    def get_text_len(hit_id):
        pair = pair_d[hit_id]
        h, p = pair
        return len(h.split(" ")), len(p.split(" "))

    grouped = group_by(all_annots, get_first)

    def enum_paired(target_sent_type_idx=None):
        for hit_id, entries in grouped.items():
            if len(entries) <= 1:
                continue
            assn1 = entries[0]
            assn2 = entries[1]
            pair = pair_d[hit_id]
            h, p = pair
            t1_len, t2_len = len(h.split(" ")), len(p.split(" "))
            _, annot1 = assn1
            _, annot2 = assn2
            for sent_type_idx, (indices1, indices2) in enumerate(zip(annot1.enum(), annot2.enum())):
                if target_sent_type_idx is not None:
                    if target_sent_type_idx != sent_type_idx:
                        continue
                sent_len = {0: t1_len,
                            1: t1_len,
                            2: t2_len,
                            3: t2_len
                            }[sent_type_idx]
                sent = {
                    0: p,
                    1: p,
                    2: h,
                    3: h
                }[sent_type_idx]
                if indices1 and max(indices1) >= sent_len:
                    print(hit_id)
                    print(sent_type_idx)
                    print(indices1)
                    print(sent)
                    print("Invalid annotation max(indices1)={} while sent_len={}".format(max(indices1), sent_len))
                    continue
                if indices2 and max(indices2) >= sent_len:
                    print(hit_id)
                    print(sent_type_idx)
                    print(indices2)
                    print(sent)
                    print("Invalid annotation max(indices2)={} while sent_len={}".format(max(indices2), sent_len))
                    continue

                for token_idx in range(sent_len):
                    b1 = token_idx in indices1
                    b2 = token_idx in indices2
                    yield b1, b2

    answer1, answer2 = zip(*enum_paired())
    print("cohens_kappa", cohens_kappa(answer1, answer2))
    for sent_type_idx in range(4):
        answer1, answer2 = zip(*enum_paired(sent_type_idx))
        print("cohens_kappa for {}".format(sent_type_idx), cohens_kappa(answer1, answer2))


def do_virtual_prec_recall(pair_d, all_annots):
    def get_text_len(hit_id):
        pair = pair_d[hit_id]
        h, p = pair
        return len(h.split(" ")), len(p.split(" "))

    grouped = group_by(all_annots, get_first)

    def enum_scores(target_sent_type_idx=None):
        for hit_id, entries in grouped.items():
            if len(entries) <= 1:
                continue
            assn1 = entries[0]
            assn2 = entries[1]
            pair = pair_d[hit_id]
            h, p = pair
            t1_len, t2_len = len(h.split(" ")), len(p.split(" "))
            _, annot1 = assn1
            _, annot2 = assn2
            for sent_type_idx, (indices1, indices2) in enumerate(zip(annot1.enum(), annot2.enum())):
                if target_sent_type_idx is not None:
                    if target_sent_type_idx != sent_type_idx:
                        continue
                tp = 0
                for token_idx in indices1:
                    if token_idx in indices2:
                        tp += 1

                num_answer = len(indices1)
                num_pred = len(indices2)
                recall = tp / num_answer if num_answer > 0 else 1
                prec = tp / num_pred if num_pred > 0 else 1
                f1 = 2*prec*recall / (prec+recall) if prec+recall > 0 else 0
                yield prec, recall, f1

    per_sent_scores = enum_scores()
    eval_str = combine_prec_recall(per_sent_scores)
    print("All\t" + eval_str)
    for sent_type_idx in range(4):
        eval_str = combine_prec_recall(enum_scores(sent_type_idx))
        print("For {}\t".format(sent_type_idx), eval_str)


def combine_prec_recall(per_sent_scores):
    l_prec, l_recall, l_f1 = zip(*per_sent_scores)
    m_prec = average(l_prec)
    m_recall = average(l_recall)
    m_f1 = average(l_f1)
    eval_str = f"{m_prec:.2f}\t{m_recall:.2f}\t{m_f1:.2f}"
    return eval_str


def main():
    all_annots = load_annots_w_processing()
    # print_annot_stats(all_annots)
    print_annot_stats_grouped(all_annots)
    pair_d = get_pair_dict()
    do_virtual_prec_recall(pair_d, all_annots)
    # do_agreement_analysis(pair_d, all_annots)



if __name__ == "__main__":
    main()