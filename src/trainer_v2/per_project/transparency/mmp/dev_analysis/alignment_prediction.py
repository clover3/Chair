from transformers import AutoTokenizer
from cache import load_from_pickle
from trainer_v2.per_project.transparency.mmp.alignment.network.alignment_predictor import compute_alignment_for_taget_q_word_id


def main():
    item = load_from_pickle("attn_grad_dev")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # input_tokens = tokenizer.convert_ids_to_tokens(item.input_ids)

    target_q_word = "when"
    target_q_word_id = tokenizer.vocab[target_q_word]
    alignment = compute_alignment_for_taget_q_word_id(item, target_q_word_id)


if __name__ == "__main__":
    main()
