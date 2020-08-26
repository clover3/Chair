from tlm.estimator_prediction_viewer import EstimatorPredictionViewerGosford
from visualize.html_visual import Cell, HtmlVisualizer


def show(filename):
    data = EstimatorPredictionViewerGosford(filename)
    html_writer = HtmlVisualizer("token_scoring.html", dark_mode=False)

    correctness = []
    for entry in data:
        tokens = entry.get_tokens("input_ids")
        logits = entry.get_vector("logits")
        masks = entry.get_vector("label_masks")
        ids = entry.get_vector("labels")



        token_row = []
        pred_row = []
        gold_row = []
        rows = [token_row, pred_row, gold_row]

        for idx, token in enumerate(tokens):
            token_cell = Cell(token)
            if token == "[PAD]":
                break
            model_score = logits[idx][0]   
            if masks[idx]:

                correct = (model_score > 0 and ids[idx] > 0) or (model_score < 0 and ids[idx] < 0)
                color = "B" if correct else "R"
                if correct and (model_score > 0 and ids[idx] > 0) :
                    color = "G"
                pred_cell = Cell("{0:.2f}".format(model_score), 100, target_color=color)
                gold_cell = Cell("{0:.2f}".format(ids[idx]), 100, target_color=color)
            else:
                token_cell = Cell(token)
                pred_cell = Cell("")
                gold_cell = Cell("")

            token_row.append(token_cell)
            pred_row.append(pred_cell)
            gold_row.append(gold_cell)

        html_writer.multirow_print_from_cells_list(rows, 20)


def main():
    show("token_scoring.pickle")


if __name__ == "__main__":
    main()