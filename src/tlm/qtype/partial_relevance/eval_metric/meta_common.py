
def higher_the_better(a, b):
    return a < b


def lower_the_better(a, b):
    return a > b


# Check if right argument is better

better_fn_d = {
    "ps_replace_precision": higher_the_better,
    "ps_replace_recall": higher_the_better,
    "deletion": higher_the_better,
    "ps_deletion_precision": higher_the_better,
    "ps_deletion_recall": higher_the_better,
    "attn_brevity": lower_the_better,
    "attn": lower_the_better,
    "attn_v3": lower_the_better,
    "replace_v3": lower_the_better,
    "erasure_v3": lower_the_better,
    "replace_v31": lower_the_better,
    "replace_v32": lower_the_better,
}


def get_better_fn(metric):
    higher = ["ps_replace_precision", "ps_replace_recall", "deletion", "ps_deletion_precision",
              "ps_deletion_recall",
              "erasure_suff_v3d", "erasure_suff_v3",
              "replace_suff_v3", "replace_suff_v3d",
              ]
    lower = [
        "attn_brevity", "attn", "attn_v3",
        "replace_v3", "replace_v31", "replace_v32", "replace_v3d",
        "erasure_v3", "erasure_v3d"]

    if metric in higher:
        return higher_the_better
    elif metric in lower:
        return lower_the_better
    else:
        raise ValueError(metric)