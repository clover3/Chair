from crs.load_stance_annotation import load_stance_annot


def gen(path):
    r = load_stance_annot(path)

    history = set()
    todo_list = []
    for e in r:
        statement = e['statement']
        for evi in [e['support_evidence'], e['dispute_evidence']]:
            if len(evi) <= 1:
                continue
            item_sig = " ".join([t.strip() for t in evi]) + statement
            print(item_sig)
            if item_sig not in history:
                todo_list.append((evi, statement))
                history.add(item_sig)





if __name__ == "__main__":
    path = "C:\work\Data\CKB annotation\Search Stances 4\\Batch_3749275_batch_results.csv"
    gen(path)



