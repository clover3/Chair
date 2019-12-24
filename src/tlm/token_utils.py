from tlm.estimator_prediction_viewer import is_dependent
from visualize.html_visual import Cell


def get_resolved_tokens_from_masked_tokens_and_ids(tokens, answer_mask_tokens, masked_positions):
    for i, t in enumerate(tokens):
        if t == "[PAD]":
            break
        if i in masked_positions:
            i_idx = masked_positions.index(i)
            tokens[i] = "[{}:{}]".format(i_idx, answer_mask_tokens[i_idx])

    return tokens



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