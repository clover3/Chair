def re_tokenize(tokens):
    out = []
    for term in tokens:
        spliter = "-"
        if spliter in term and term.find(spliter) > 0:
            out.extend(term.split(spliter))
        else:
            out.append(term)
    return set(out)