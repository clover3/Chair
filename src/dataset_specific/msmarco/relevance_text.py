import numpy as np

from misc_lib import two_digit_float


def make_relevance_prediction_summary_str(base_probs):
    pred = np.argmax(base_probs)
    orignal_prediction_str = ['Non-relevant', 'Relevant'][pred]
    original_prediction_summary = "{} ({}, {})".format(orignal_prediction_str,
                                                           two_digit_float(base_probs[0]),
                                                           two_digit_float(base_probs[1]),
                                                           )
    return original_prediction_summary