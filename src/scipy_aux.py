import scipy.special


def logit_to_score_softmax(logit):
    return scipy.special.softmax(logit)[1]