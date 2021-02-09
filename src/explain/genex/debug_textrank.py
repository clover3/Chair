from cpath import at_output_dir, at_data_dir
from explain.genex.load import load_as_tokens
from misc_lib import get_f1, NamedAverager


def load(file_path, cut):
    l = []
    for line in open(file_path, "r"):
        terms = line.split()
        terms = terms[:cut]
        l.append(terms)
    return l



def main():
    # run1 = load(at_output_dir("genex", "textrank.txt"), 3)
    # run2 = load(at_output_dir("genex", "textrank-ts.txt"), 3)
    problems = load_as_tokens("tdlt")
    run1 = load(at_output_dir("genex", "1"), 3)
    run2 = load(at_output_dir("genex", "2_ts"), 3)
    gold = load(at_data_dir("genex", "tdlt_gold.txt"), 999)

    def common(pred, gold):
        return list([t for t in pred if t in gold])

    n_correct_1 = 0
    n_correct_2 = 0
    
    d1 = NamedAverager()
    d2 = NamedAverager()

    for idx, (t1, t2, t_gold, problem) in enumerate(zip(run1, run2, gold, problems)):
        c1 = common(t1, t_gold)
        c2 = common(t2, t_gold)

        p1 = len(c1) / len(t1)
        r1 = len(c1) / len(t_gold)
        f1 = get_f1(p1, r1)
        d1['prec'].append(p1)
        d1['recall'].append(r1)
        d1['f1'].append(f1)
        
        p2 = len(c2) / len(t2)
        r2 = len(c2) / len(t_gold)
        f2 = get_f1(p2, r2)
        d2['prec'].append(p2)
        d2['recall'].append(r2)
        d2['f1'].append(f2)

        n_correct_1 += len(c1)
        n_correct_2 += len(c2)

        if len(c1) != len(c2):
            print()
            print(">> Problem ", idx)
            print("textrank :", c1)
            print("textrank-ts :", c2)

            q_match = len(common(problem.query, problem.doc))
            n_q = len(problem.query)
            if len(c1) < len(c2):
                d2['q_match_rate'].append(q_match/n_q)
            else:
                d1['q_match_rate'].append(q_match / n_q)
            print('query: ', problem.query)
            print("matching query terms: ", common(problem.query, problem.doc))
            print('doc: ', " ".join(problem.doc))

    print("{} vs {}".format(n_correct_1, n_correct_2))

    print(d1.get_average_dict())
    print(d2.get_average_dict())

    print(d1.avg_dict['q_match_rate'].history)
    print(d2.avg_dict['q_match_rate'].history)

if __name__ == "__main__":
    main()