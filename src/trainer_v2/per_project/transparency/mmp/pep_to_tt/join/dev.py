from table_lib import tsv_iter
from trainer_v2.per_project.transparency.misc_common import load_table
import scipy.stats


def main():
    path1 = "output/mmp/tables/car_pep14_20K.tsv"
    path2 = "output/mmp/mct6_pep_tt21_40000/400.txt"

    print(path1)
    table1 = load_table(path1)
    print(path2)
    entries2 = {k: float(v) for k, v in tsv_iter(path2)}

    table = []
    for q_term, entries in table1.items():
        entries1 = list(entries.items())
        entries1.sort(key=lambda x: x[1], reverse=True)
        for d_term, score in entries1:
            if d_term in entries2:
                row = [q_term, d_term, score, entries2[d_term]]
                table.append(row)

    x = [row[2] for row in table]
    y = [row[3] for row in table]

    ret = scipy.stats.linregress(x, y)
    a = ret.slope
    b = ret.intercept

    for qt, dt, s1, s2 in table:
        s1p = s1 * a + b
        print(qt, dt, s1, s1p, s2)



if __name__ == "__main__":
    main()