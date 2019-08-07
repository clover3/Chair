

def parse_title_line(line):
    pre_n = len("<DOCNO>")
    ed_n = len("</DOCNO>") + 1
    title = line[pre_n:-ed_n].strip()
    tokens = title.split("-")
    st, ed = tokens[-2], tokens[-1]
    title_only = "-".join(tokens[:-2])
    return title_only, int(st), int(ed)
