import numpy as np

from cache import load_from_pickle
from data_generator.NLI import nli
from data_generator.NLI.nli import get_modified_data_loader2
from data_generator.common import get_tokenizer
from explain.ex_train_modules import NLIExTrainConfig
from models.transformer.hyperparams import HPSENLI3
from tlm.token_utils import cells_from_tokens
from visualize.html_visual import HtmlVisualizer, Cell, set_cells_color
from visualize.tex_visualizer import TexNLIVisualizer, TexTableNLIVisualizer


def normalize(scores):
    max_score = max(scores)
    min_score = min(scores)

    gap = max_score - min_score
    if gap < 0.001:
        gap = 1

    return [(s - min_score) / gap * 100 for s in scores]


def binarize(arr, t, true_val, false_val=0):
    return [true_val if v > t else false_val for v in arr]


def show_all(run_name, data_id):
    num_tags = 3
    num_select = 1
    pickle_name = "save_view_{}_{}".format(run_name, data_id)
    tokenizer = get_tokenizer()

    data_loader = get_modified_data_loader2(HPSENLI3(), NLIExTrainConfig())

    explain_entries = load_from_pickle(pickle_name)
    explain_entries = explain_entries

    visualizer = HtmlVisualizer(pickle_name + ".html")
    tex_visulizer = TexTableNLIVisualizer(pickle_name + ".tex")
    tex_visulizer.begin_table()
    selected_instances = [[], [], []]
    for idx, entry in enumerate(explain_entries):
        x0, logits, scores = entry

        pred = np.argmax(logits)
        input_ids = x0
        p, h = data_loader.split_p_h_with_input_ids(input_ids, input_ids)
        p_tokens = tokenizer.convert_ids_to_tokens(p)
        h_tokens = tokenizer.convert_ids_to_tokens(h)

        p_rows = []
        h_rows = []
        p_rows.append(cells_from_tokens(p_tokens))
        h_rows.append(cells_from_tokens(h_tokens))

        p_score_list = []
        h_score_list = []
        for j in range(num_tags):
            tag_name = nli.tags[j]
            p_score, h_score = data_loader.split_p_h_with_input_ids(scores[j], input_ids)
            normalize_fn = normalize

            add = True
            if pred == "0":
                add = tag_name == "match"
            if pred == "1":
                add = tag_name == "mismatch"
            if pred == "2":
                add = tag_name == "conflict"

            def format_scores(raw_scores):
                def format_float(s):
                    return "{0:.2f}".format(s)

                norm_scores = normalize_fn(raw_scores)

                cells = [Cell(format_float(s1), s2, False, False) for s1, s2 in zip(raw_scores, norm_scores)]
                if tag_name == "mismatch":
                    set_cells_color(cells, "G")
                elif tag_name == "conflict":
                    set_cells_color(cells, "R")
                return cells

            if add:
                p_rows.append(format_scores(p_score))
                h_rows.append(format_scores(h_score))

            p_score_list.append(p_score)
            h_score_list.append(h_score)

        pred_str = ["Entailment", "Neutral" , "Contradiction"][pred]

        out_entry = pred_str, p_tokens, h_tokens, p_score_list, h_score_list

        if len(selected_instances[pred]) < num_select :
            selected_instances[pred].append(out_entry)
            visualizer.write_headline(pred_str)
            visualizer.multirow_print_from_cells_list(p_rows)
            visualizer.multirow_print_from_cells_list(h_rows)
            visualizer.write_instance(pred_str, p_rows, h_rows)

            tex_visulizer.write_paragraph(str(pred))
            tex_visulizer.multirow_print_from_cells_list(p_rows, width=13)
            tex_visulizer.multirow_print_from_cells_list(h_rows, width=13)

        if all([len(s) == num_select for s in selected_instances]):
            break

    tex_visulizer.close_table()
    return selected_instances


def apply_color(cells, tag_name):
    if tag_name == "mismatch":
        set_cells_color(cells, "G")
    elif tag_name == "conflict":
        set_cells_color(cells, "R")
    return cells


def restore_capital_letter(tokens):
    cap_list = ["Jobs", "Texas", "Lewisville", "Tuppence"]
    t = "Satyajit Ray Mrinal Sen shown in Europe or North America than in India itself of Mrinal Sen work can be found in European collections"
    for token in t.split():
        if token[0].isupper():
            cap_list.append(token)
    tokens[0] = tokens[0].capitalize()

    cap_list = set([t.lower() for t in cap_list])

    for i in range(1, len(tokens)):
        if tokens[i] in cap_list:
            tokens[i] = tokens[i].capitalize()

    return tokens


def show_simple(run_name, data_id, tex_visulizer):
    num_tags = 3
    num_select = 20
    pickle_name = "save_view_{}_{}".format(run_name, data_id)
    tokenizer = get_tokenizer()

    data_loader = get_modified_data_loader2(HPSENLI3(), NLIExTrainConfig())

    explain_entries = load_from_pickle(pickle_name)
    explain_entries = explain_entries

    selected_instances = [[], [], []]
    for idx, entry in enumerate(explain_entries):
        x0, logits, scores = entry

        pred = np.argmax(logits)
        input_ids = x0
        p, h = data_loader.split_p_h_with_input_ids(input_ids, input_ids)
        p_tokens = tokenizer.convert_ids_to_tokens(p)
        h_tokens = tokenizer.convert_ids_to_tokens(h)

        p_tokens = restore_capital_letter(p_tokens)
        h_tokens = restore_capital_letter(h_tokens)

        target_tag = ["match", "mismatch", "conflict"][pred]

        tag_idx = nli.tags.index(target_tag)
        tag_name = nli.tags[tag_idx]
        p_score, h_score = data_loader.split_p_h_with_input_ids(scores[tag_idx], input_ids)
        normalize_fn = normalize
        p_score = normalize_fn(p_score)
        h_score = normalize_fn(h_score)
        p_row = [Cell("\\textbf{P:}")] + cells_from_tokens(p_tokens, p_score)
        h_row = [Cell("\\textbf{H:}")] + cells_from_tokens(h_tokens, h_score)

        pred_str = ["entailment", "neutral" , "contradiction"][pred]
        apply_color(p_row, tag_name)
        apply_color(h_row, tag_name)
        #tex_visulizer.write_paragraph(pred_str)
        if len(selected_instances[pred]) < num_select :
            e = pred_str, [p_row, h_row]
            #tex_visulizer.write_instance(pred_str, gold_label, [p_row, h_row])
            selected_instances[pred].append(e)

        if all([len(s) == num_select for s in selected_instances]):
            break

    for insts in selected_instances:
        for e in insts:
            pred_str, rows = e
            tex_visulizer.write_instance(pred_str, rows)

    return selected_instances

def run_show_all():
    run_name = "nli_ex_19"
    for tag in ["conflict", "match", "mismatch"]:
        show_all(run_name, "{}_1000".format(tag))


def run_show_simple():
    run_name = "nli_ex_19"
    #run_name = "nli_ex_26"
    tex_visulizer = TexNLIVisualizer("20_example.tex")
    for tag in ["conflict", "match", "mismatch"]:
        tex_visulizer.begin_table()
        show_simple(run_name, "{}_1000".format(tag), tex_visulizer)
        tex_visulizer.close_table()


if __name__ == "__main__":
    run_show_simple()