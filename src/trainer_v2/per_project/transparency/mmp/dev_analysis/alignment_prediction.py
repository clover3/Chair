from transformers import AutoTokenizer
import tensorflow as tf
from cache import load_from_pickle
from trainer_v2.per_project.transparency.mmp.alignment.grad_extractor import ModelEncoded
from typing import List, Iterable, Callable, Dict, Tuple, Set
import numpy as np


def get_seg2_start(token_type_ids):
    for i in range(len(token_type_ids)):
        if token_type_ids[i] == 1:
            return i
    return len(token_type_ids)


def get_idx_of(input_ids, token_id):
    for idx, t in enumerate(input_ids):
        if t == token_id:
            return idx
    return -1


def compute_alignment(
        item: ModelEncoded,
        target_q_word_id: int,
        top_k=10,
) -> List[Tuple[int, int, float]]:
    d_start = get_seg2_start(item.token_type_ids)
    idx = get_idx_of(item.input_ids, target_q_word_id)
    # print("Target is {} at {} th".format(input_tokens[idx], idx))
    # print("input_tokens", input_tokens)
    assert item.token_type_ids[idx] == 0

    attn_mul_grad = item.attentions * item.attention_grads

    t = tf.reduce_sum(attn_mul_grad, axis=0)  # average over layers
    t = tf.reduce_sum(t, axis=0)  # average over heads
    t = t[:, idx] + t[idx, :]  #
    # t = t[idx, :] #
    rank = np.argsort(t)[::-1]
    rank = [j for j in rank if item.input_ids[j] != 0 and j >= d_start]
    output: List[Tuple[int, int, float]] = []
    for j in rank[:top_k]:
        score = t[j].numpy().tolist()
        output.append((int(j), int(item.input_ids[j]), float(score)))
    return output


def main():
    item = load_from_pickle("attn_grad_dev")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # input_tokens = tokenizer.convert_ids_to_tokens(item.input_ids)

    target_q_word = "when"
    target_q_word_id = tokenizer.vocab[target_q_word]
    alignment = compute_alignment(item, target_q_word_id)


if __name__ == "__main__":
    main()
