

def find_sent(doc, sent):
    idx = doc.find(sent)
    if idx >= 0 :
        return idx, idx + len(sent)

    raise NotImplementedError()

