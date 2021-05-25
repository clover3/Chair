import scipy.special


def logit_to_score_softmax(logit):
    return scipy.special.softmax(logit)[1]


def get_logits_to_score_fn(score_type):
    if score_type == "softmax":
        def get_score(logits):
            return logit_to_score_softmax(logits)
    elif score_type == "raw":
        def get_score(logits):
            return logits[0]
    elif score_type == "scalar":
        def get_score(logits):
            return logits
    elif score_type == "tuple":
        def get_score(logits):
            return logits[1]
    else:
        assert False

    return get_score