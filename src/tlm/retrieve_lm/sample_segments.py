from misc_lib import TimeEstimator
from tlm.retrieve_lm.mysql_sentence import *
from tlm.retrieve_lm.select_sentence import get_random_sent
from cache import StreamPickler
from data_generator import tokenizer_b as tokenization
import path
import os


def extend(doc_rows, sent, loc, tokenizer, max_seq):
    loc_st = loc
    loc_ed = loc + 1

    num_rows = len(doc_rows)
    allow = 3 # a sentence with less than 3 tokens might not be meaningful

    FORWARD = 1
    BACKWARD = 0
    direction = FORWARD
    last_direction = BACKWARD
    sent_list = [sent]
    tokenize = tokenizer.tokenize
    target_tokens = tokenizer.tokenize(sent)

    def sent_at(loc):
        return doc_rows[loc][4]

    while len(target_tokens) + allow < max_seq:
        # Add a sentence
        if last_direction == BACKWARD:
            if loc_st-1 > 0:
                direction = FORWARD
            elif loc_ed +1 <= num_rows:
                direction = BACKWARD
            else:
                # Cannot extend anymore
                break
        else:
            if loc_ed +1 <= num_rows:
                direction = BACKWARD
            elif loc_st-1 > 0:
                direction = FORWARD
            else:
                break

        if direction == FORWARD:
            new_sent = sent_at(loc_st-1)
            new_tokens = tokenize(new_sent)
            if len(target_tokens) + len(new_tokens) > max_seq:
                break

            sent_list = [new_sent] + sent_list
            target_tokens = new_tokens + target_tokens
            loc_st -= 1
        elif direction == BACKWARD:
            new_sent = sent_at(loc_ed)
            new_tokens = tokenize(new_sent)
            if len(target_tokens) + len(new_tokens) > max_seq:
                break
            sent_list = sent_list + [new_sent]
            target_tokens = target_tokens + new_tokens
            loc_ed += 1
        else:
            assert False
        last_direction = direction

    prev_tokens = []
    for i in range(0, loc_st):
        prev_tokens.extend(tokenize(sent_at(i)))
    prev_tokens = prev_tokens[-max_seq:]

    next_tokens = []
    for i in range(loc_ed, num_rows):
        next_tokens.extend(tokenize(sent_at(i)))
    next_tokens = next_tokens[:max_seq]
    return target_tokens, sent_list, prev_tokens, next_tokens



def visualize_inst(inst):
    target_tokens, sent_list, prev_tokens, next_tokens = inst

    print("< Num tokens > ")
    print("target_tokens : ", len(target_tokens))
    print("prev_tokens : ", len(prev_tokens))
    print("next_tokens : ", len(next_tokens))



def main():
    num_inst = 1000 * 1000
    max_seq = 256
    vocab_file = os.path.join(path.data_path, "bert_voca.txt")
    tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)

    sp = StreamPickler("robust_segments_", 1000)
    ticker = TimeEstimator(num_inst)
    for i in range(num_inst):
        r = get_random_sent()
        s_id, doc_id, loc, g_id, sent= r
        doc_rows = get_doc_sent(doc_id)
        target_tokens, sent_list, prev_tokens, next_tokens = extend(doc_rows, sent, loc, tokenizer, max_seq)

        inst = target_tokens, sent_list, prev_tokens, next_tokens, doc_id

        sp.add(inst)
        ticker.tick()

    sp.flush()


if __name__ == "__main__":
    main()
