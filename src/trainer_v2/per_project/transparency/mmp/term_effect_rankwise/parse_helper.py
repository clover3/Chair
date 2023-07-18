from dataset_specific.msmarco.passage.passage_resource_loader import tsv_iter


def read_qid_pid_score_tsv(qid, save_path):
    entries = []
    for qid, pid, score in tsv_iter(save_path):
        entries.append((pid, float(score)))
    return qid, entries


def escape(part_of_name):
    todo  = [
        ("#", "[SHARP]"),
        ("*", "[ASTERISK]"),
        ("\\", "[BKSLASH]"),
        ("/", "[SLASH]"),
        ("%", "[PERCENT]"),
        ("?", "[QUESTIONMARK]"),
        (";", "[SEMICOLON]"),
        ("|", "[PIPE]"),
        ("@", "[AT]"),
        ("$", "[DOLLAR]"),
        ("`", "[BACKTICK]"),
    ]
    name = part_of_name
    for ch, text in todo:
        name = name.replace(ch, text)
    return name