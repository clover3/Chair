from crs.load_stance_annotation import load_stance_verify_annot
from collections import Counter
from misc_lib import right
from stats.agreement import binary_kappa



def combined_agreement(path):
    data = load_stance_verify_annot(path)
    group = {}
    sig2data = {}
    for e in data:
        sig = e['statement'] + e['link']
        sig2data[sig] = e['statement'], e['link']
        if sig not in group:
            group[sig] = []

        group[sig].append((e['support'], e['dispute']))

    NOT_FOUND = 0
    YES = 1
    NOT_SURE = 2


    def get_cont_annot(annot_idx):
        statement_group = {}
        cont_annot = []
        for sig in group:
            statement, link = sig2data[sig]
            s, d = group[sig][annot_idx]
            if statement not in statement_group:
                statement_group[statement] = []
            statement_group[statement].append((link, s, d))

        for statement in statement_group:
            s_yes_cnt = 0
            d_yes_cnt = 0

            for link, s, d in statement_group[statement]:
                if s == YES:
                    s_yes_cnt += 1
                if d == YES:
                    d_yes_cnt += 1
            if s_yes_cnt > 0 and d_yes_cnt > 0:
                cont = True
            else:
                cont = False
            cont_annot.append((statement, cont))
        return cont_annot
    annot1 = get_cont_annot(0)
    annot2 = get_cont_annot(1)

    annot1.sort(key=lambda x:x[0])
    annot2.sort(key=lambda x: x[0])

    for e1,e2 in zip(annot1, annot2):
        assert e1[0] == e2[0]

    kappa, p0 = binary_kappa(right(annot1),right(annot2))
    print("kappa", kappa)
    print("p0", p0)


def merge(path):
    data = load_stance_verify_annot(path)

    group = {}
    sig2data = {}
    for e in data:
        sig = e['statement'] + e['link']
        sig2data[sig] = e['statement'], e['link']
        if sig not in group:
            group[sig] = []

        group[sig].append((e['support'], e['dispute']))

    NOT_FOUND = 0
    YES = 1
    NOT_SURE = 2

    statement_group = {}

    for sig in group:
        statement, link = sig2data[sig]

        s_yes_cnt = 0
        s_no_cnt = 0
        d_yes_cnt = 0
        d_no_cnt = 0
        for s, d in group[sig]:
            if s == YES:
                s_yes_cnt += 1
            elif s == NOT_FOUND:
                s_no_cnt += 1

            if d == YES:
                d_yes_cnt += 1
            elif d == NOT_FOUND:
                d_no_cnt += 1

        s_conclusion = 0
        assert s_yes_cnt + s_no_cnt <= 3
        assert d_yes_cnt + d_no_cnt <= 3
        if s_yes_cnt > 1.5 :
            s_conclusion = YES
        elif s_no_cnt > 1.5:
            s_conclusion = NOT_FOUND
        else:
            s_conclusion = NOT_SURE

        d_conclusion = NOT_SURE
        if d_yes_cnt > 1.5:
            d_conclusion = YES
        elif d_no_cnt > 1.5:
            d_conclusion = NOT_FOUND
        else:
            d_conclusion = NOT_SURE

        if statement not in statement_group:
            statement_group[statement] = []

        statement_group[statement].append((link, s_conclusion, d_conclusion))

    CONTROVERSIAL = 1
    NOT_CONTROVERSIAL = 0
    NOT_SURE = 2

    stat = Counter()
    for statement, evidences in statement_group.items():
        n_support = 0
        n_dispute = 0

        n_no_support =0
        n_no_dispute =0

        for e in evidences:
            link, support, dispute = e
            if support == YES:
                n_support += 1
            if dispute == YES:
                n_dispute += 1

            if support == NOT_FOUND:
                n_no_support += 1
            if dispute == NOT_FOUND:
                n_no_dispute += 1

        conclusion = "Not Sure"
        if n_support > 0 and n_dispute > 0:
            conclusion = "Controversial"
        elif n_no_dispute == len(evidences) or n_no_support == len(evidences):
            conclusion = "Not controversial"

        stat[conclusion] += 1

        print(statement, conclusion)

    for k,v in stat.items():
        print(k, v)


if __name__ == "__main__":
    path = "C:\work\Code\Chair\data\crs\\verify\\-1.csv"
    merge(path)



