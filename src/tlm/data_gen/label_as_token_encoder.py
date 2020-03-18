from list_lib import foreach


def get_label_token(label):
    idx = 10 + label
    return "[unused{}]".format(idx)


def get_unk_label_token():
    return "[unused9]"


def encode_label_and_token_pair(topic_tokens, label, tokens_labeled, tokens_unlabeled, swap):
    tokens = []
    segment_ids = []
    cur_segment_type = 0

    label_token = get_label_token(label)
    sent1 = tokens_labeled if not swap else tokens_unlabeled
    label_1 = label_token if not swap else get_unk_label_token()
    sent2 = tokens_unlabeled if not swap else tokens_labeled
    label_2 = get_unk_label_token() if not swap else label_token

    def append_token(token):
        tokens.append(token)
        segment_ids.append(cur_segment_type)

    append_token("[CLS]")
    foreach(append_token, topic_tokens)
    append_token(label_1)
    foreach(append_token, sent1)
    append_token("[SEP]")

    cur_segment_type = 1
    foreach(append_token, topic_tokens)
    append_token(label_2)
    foreach(append_token, sent2)
    append_token("[SEP]")
    return tokens, segment_ids

