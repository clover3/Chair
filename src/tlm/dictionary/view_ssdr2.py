import os
import pickle

from cpath import output_path
from data_generator.tokenizer_wo_tf import get_tokenizer
from tlm.token_utils import get_resolved_tokens_from_masked_tokens_and_ids, cells_from_tokens
from visualize.html_visual import HtmlVisualizer, Cell


def do():

    data = pickle.load(open(os.path.join(output_path, "ssdr.pickle"), "rb"))

    max_seq_length = 512
    batch_size = 256
    inner_batch_size = 32
    n_machine = 8
    max_predictions_per_seq = 20
    def_per_batch = 32 * 10

    html = HtmlVisualizer("ssdr.html")
    tokenizer = get_tokenizer()
    def is_multitoken(d_location_ids):
        for i, p in enumerate(d_location_ids):
            if p == 0:
                break
            if d_location_ids[i+1] == p+1:
                return True
        return False



    for entry in data:
        for machine_idx in range(n_machine):
            local_ab_mapping = entry["ab_mapping"][machine_idx]
            st = machine_idx * def_per_batch
            ed = (machine_idx +1) * def_per_batch
            local_scores = entry["scores"][st:ed]
            ab_reverse = [list() for _ in range(inner_batch_size)]
            print(local_ab_mapping)
            no_more_zero = False
            for def_idx, target_idx in enumerate(local_ab_mapping):
                if no_more_zero and target_idx == 0:
                    break
                ab_reverse[target_idx].append(def_idx)
                if target_idx > 0:
                    no_more_zero = True

            local_batch = []
            for i in range(inner_batch_size):
                a_idx = i + machine_idx * inner_batch_size
                input_ids = entry["masked_input_ids"][a_idx]
                masked_lm_ids = entry["masked_lm_ids"][a_idx]
                masked_lm_positions = entry["masked_lm_positions"][a_idx]
                d_location_ids = entry["d_location_ids"][a_idx]
                local_batch.append((input_ids, masked_lm_ids, masked_lm_positions))

                tokens = get_resolved_tokens_from_masked_tokens_and_ids(
                    tokenizer.convert_ids_to_tokens(input_ids),
                    tokenizer.convert_ids_to_tokens(masked_lm_ids),
                    list(masked_lm_positions))
                html.write_headline("Document {}-{}".format(machine_idx, i))
                if is_multitoken(d_location_ids):
                    html.write_paragraph("Multi-token")

                cells = cells_from_tokens(tokens)
                for j, c in enumerate(cells):
                    if j != 0 and j in d_location_ids:
                        cells[j].highlight_score = 50

                html.multirow_print(cells)
                html.write_headline("Dictionary")
                row_list = []
                max_score = -9999
                max_idx = 0
                for def_idx in ab_reverse[i]:
                    score = local_scores[def_idx]
                    if max_score < score:
                        max_score = score
                        max_idx = def_idx

                for def_idx in ab_reverse[i]:
                    g_def_idx = def_idx + machine_idx * def_per_batch
                    tokens = tokenizer.convert_ids_to_tokens(entry["d_input_ids"][g_def_idx])
                    score = local_scores[def_idx]
                    if def_idx == max_idx:
                        row = [Cell(def_idx), Cell(score, 100)]
                    else:
                        row = [Cell(def_idx), Cell(score)]
                    row += cells_from_tokens(tokens)
                    row_list.append(row)

                for row in row_list:
                    html.write_table([row])

        break

if __name__ == "__main__":
    do()