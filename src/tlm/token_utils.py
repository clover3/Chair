from visualize.html_visual import Cell


def get_resolved_tokens_from_masked_tokens_and_ids(tokens, answer_mask_tokens, masked_positions):
    for i, t in enumerate(tokens):
        if t == "[PAD]":
            break
        if i in masked_positions:
            i_idx = masked_positions.index(i)
            tokens[i] = "[{}:{}]".format(i_idx, answer_mask_tokens[i_idx])

    return tokens

def mask_resolve_1(tokens, masked_tokens):
    for i in range(len(tokens)):
        if masked_tokens[i] == "[MASK]":
            masked_tokens[i] = "[{}]".format(tokens[i])
        if tokens[i] == "[SEP]":
            tokens[i] = "[SEP]<br>"

    return masked_tokens


def get_resolved_tokens_by_mask_id(tokenizer, feature):
    masked_inputs = feature["input_ids"].int64_list.value
    tokens = tokenizer.convert_ids_to_tokens(masked_inputs)
    mask_tokens = tokenizer.convert_ids_to_tokens(feature["masked_lm_ids"].int64_list.value)
    masked_positions = list(feature["masked_lm_positions"].int64_list.value)
    print(masked_positions)

    for i, t in enumerate(tokens):
        if t == "[PAD]":
            break
        if i in masked_positions:
            i_idx = masked_positions.index(i)
            tokens[i] = "[{}:{}]".format(i_idx, mask_tokens[i])

    return tokens


def is_mask(token):
    if len(token) > 2 and token[-1] == "]" and token[-2].islower():
        return 100
    else:
        return 0


def cells_from_scores(scores, hightlight=True):
    cells = []
    for i, score in enumerate(scores):
        h_score = score if hightlight else 0
        cells.append(Cell(float_aware_strize(score), h_score))
    return cells


def cells_from_tokens(tokens, scores=None, stop_at_pad=True):
    cells = []
    for i, token in enumerate(tokens):
        if tokens[i] == "[PAD]" and stop_at_pad:
            break
        term = tokens[i]
        cont_left = term[:2] == "##"
        cont_right = i + 1 < len(tokens) and tokens[i + 1][:2] == "##"
        if i + 1 < len(tokens):
            dependent_right = is_dependent(tokens[i + 1])
        else:
            dependent_right = False

        dependent_left = is_dependent(tokens[i])

        if cont_left:
            term = term[2:]

        space_left = "&nbsp;" if not (cont_left or dependent_left) else ""
        space_right = "&nbsp;" if not (cont_right or dependent_right) else ""

        if scores is not None:
            score = scores[i]
        else:
            score = 0
        cells.append(Cell(term, score, space_left, space_right))
    return cells


def is_dependent(token):
    return len(token) == 1 and not token[0].isalnum()


def float_aware_strize(obj):
    try:
        i = int(obj)
        v = float(obj)

        if abs(i-v) < 0.0001: # This is int
            return str(obj)
        else:
            return "{:04.2f}".format(obj)
    except:
        return str(obj)

    return str(v)