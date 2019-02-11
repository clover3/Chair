
ENTAILMENT = 0
NEUTRAL = 1
CONTRADICTION = 2


def get_target_class(explain_tag):
    target_class = {
        'conflict': CONTRADICTION,
        'match': ENTAILMENT,
        'dontcare': ENTAILMENT,
        'mismatch': NEUTRAL,
    }[explain_tag]
    return target_class