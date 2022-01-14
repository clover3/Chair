from typing import List


def seg_contribution_from_scores(
        base_score: float,
        single_sent_result: List[float],
        sent_drop_result: List[float]):
    single_sent_based_score = [after for after in single_sent_result]
    drop_sent_based_score = contribution_by_change(base_score, sent_drop_result)
    avg_sent_score = [(a+b)/2 for a, b in zip(single_sent_based_score, drop_sent_based_score)]
    return avg_sent_score


def contribution_by_change(base_score, sent_drop_result):
    drop_sent_based_score = [base_score - after for after in sent_drop_result]
    return drop_sent_based_score


def enum_window_drop(tokens, unknown_token, window_size):
    cursor = 0
    while cursor < len(tokens):
        ed = cursor + window_size
        head = tokens[:cursor]
        tail = tokens[ed:]
        out_tokens = head + [unknown_token] + tail
        yield out_tokens
        cursor = ed