from list_lib import get_max_idx

ENTAILMENT = 0
NEUTRAL = 1
CONTRADICTION = 2

MARK_PREM = 2
MARK_HYPO = 3
nli_label_list = ["entailment", "neutral", "contradiction", ]
enli_tags = ["match", "mismatch",  "conflict"]


def get_target_class(explain_tag):
    target_class = {
        'conflict': CONTRADICTION,
        'match': ENTAILMENT,
        'dontcare': ENTAILMENT,
        'mismatch': NEUTRAL,
    }[explain_tag]
    return target_class


def get_target_class_set(explain_tag):
    target_class = {
        'conflict': [CONTRADICTION],
        'match': [ENTAILMENT, CONTRADICTION, NEUTRAL],
        'dontcare': ENTAILMENT,
        'mismatch': [NEUTRAL],
    }[explain_tag]
    return target_class


def get_segment_marker(segment_id):
    if segment_id == 0:
        return MARK_PREM
    elif segment_id == 1:
        return MARK_HYPO
    else:
        assert False


def nli_probs_str(probs):
    label = get_max_idx(probs)
    return "{0}{1:.2f}".format(nli_label_list[label][0], probs[label])


snli_train_size = 550152
mnli_train_size = 392702
prefix_d = {
    "match": "e",
    "mismatch": "n",
    "conflict": "c",
}
mnli_ex_tags = ["match", "mismatch", "conflict"]


def is_mnli_ex_target(tag_type, sent_type):
    assert sent_type in ["prem", "hypo"]
    if tag_type == "conflict":
        return True
    elif tag_type == "match":
        if sent_type == "prem":
            return True
        else:
            return False
    elif tag_type == "mismatch":
        if sent_type == "hypo":
            return True
        else:
            return False
    return