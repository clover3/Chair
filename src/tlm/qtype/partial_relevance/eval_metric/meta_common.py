
def higher_the_better(a, b):
    return a < b


def lower_the_better(a, b):
    return a > b


better_fn_d = {
    "ps_replace_precision": higher_the_better,
    "ps_replace_recall": higher_the_better,
    "deletion": higher_the_better,
    "ps_deletion_precision": higher_the_better,
    "ps_deletion_recall": higher_the_better,
    "attn_brevity": lower_the_better,
    "attn": lower_the_better,
}