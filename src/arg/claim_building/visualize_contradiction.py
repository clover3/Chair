from cache import StreamPickleReader
from data_generator.bert_input_splitter import split_p_h_with_input_ids
from data_generator.tokenizer_wo_tf import get_tokenizer
from tlm.token_utils import cells_from_tokens
from visualize.html_visual import HtmlVisualizer, normalize, Cell


def run():
    tokenizer = get_tokenizer()
    spr = StreamPickleReader("contradiction_prediction")

    html = HtmlVisualizer("contradiction_prediction.html")
    cnt = 0
    while spr.has_next():
        item = spr.get_item()
        e, p = item
        input_ids, _, _ = e
        logit, explain = p
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        p, h = split_p_h_with_input_ids(tokens, input_ids)
        p_score, h_score = split_p_h_with_input_ids(explain, input_ids)

        p_score = normalize(p_score)
        h_score = normalize(h_score)
        p_cells = [Cell("P:")] + cells_from_tokens(p, p_score)
        h_cells = [Cell("H:")] + cells_from_tokens(h, h_score)

        html.write_paragraph(str(logit))
        html.multirow_print(p_cells)
        html.multirow_print(h_cells)

        if cnt > 100:
            break
        cnt += 1


if __name__ == "__main__":
    run()
