import random
from cache import *


def generate_mask(inst, max_num_tokens, masked_lm_prob, short_seq_prob, rng):
    max_predictions_per_seq = 20

    target_tokens, sent_list, prev_tokens, next_tokens = inst

    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)
        short_seg = target_tokens[:target_seq_length]
        remain_seg = target_tokens[target_seq_length:]
        next_tokens = (remain_seg + next_tokens)[:max_num_tokens]
        target_tokens = short_seg


    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(target_tokens) * masked_lm_prob))))

    cand_indice = list(range(0, len(target_tokens)))
    rng.shuffle(cand_indice)
    mask_indice = cand_indice[:num_to_predict]

    mask_inst = target_tokens, sent_list, prev_tokens, next_tokens, mask_indice
    return mask_inst


def main():
    rng = random.Random(0)
    max_num_tokens = 256
    masked_lm_prob = 0.15
    short_seq_prob = 0.1

    spr = StreamPickleReader("robust_segments_")
    sp = StreamPickler("robust_problems_", 1000)

    while spr.has_next():
        inst = spr.get_item()
        mask_inst = generate_mask(inst, max_num_tokens, masked_lm_prob, short_seq_prob, rng)
        sp.add(mask_inst)
    sp.flush()


if __name__ == "__main__":
    main()
