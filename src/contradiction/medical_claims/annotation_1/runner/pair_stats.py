from contradiction.medical_claims.annotation_1.load_data import load_alamri1_all


def main():
    data = load_alamri1_all()

    for group_no, pairs in data:
        group_max_len = -1
        for t1, t2 in pairs:
            t1_len = len(t1.split())
            t2_len = len(t2.split())
            n_tokens = t1_len + t2_len
            bert_rep_length = n_tokens * 2 + 3

            group_max_len = max(group_max_len, bert_rep_length)

        print(group_no, group_max_len)
"""
1 123
2 105
3 145
4 167
5 143
6 83
7 111
8 135
9 127
10 185
11 157
12 155
13 143
14 143
15 151
16 113
17 109
18 167
19 155
20 145
21 133
22 159
23 125
24 125
"""

if __name__ == "__main__":
    main()