
ENTAILMENT = 0
NEUTRAL = 1
CONTRADICTION = 2

MARK_PREM = 2
MARK_HYPO = 3

def get_target_class(explain_tag):
    target_class = {
        'conflict': CONTRADICTION,
        'match': ENTAILMENT,
        'dontcare': ENTAILMENT,
        'mismatch': NEUTRAL,
    }[explain_tag]
    return target_class


def get_segment_marker(segment_id):
    if segment_id == 0:
        return MARK_PREM
    elif segment_id == 1:
        return MARK_HYPO
    else:
        assert False