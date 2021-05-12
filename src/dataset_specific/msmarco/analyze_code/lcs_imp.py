import time

from list_lib import get_max_idx, left, right


def lcs(text_a, text_b, debug=False):
    if not text_a or not text_b:
        return 0, []
    a_len = len(text_a)
    b_len = len(text_b)

    lcs_val = {}
    # lcs_arr[i,j] = length of lcs of a[:i], b[:j]

    # lcs_arr[i,j] = max(
    #   1) lcs_arr[i-1, j-1] + 1  if a[i]==b[j]
    #   2) lcs_arr[i-1, j]
    #   3) lcs_arr[i, j-1]

    lcs_type = {}

    def get_arr_val(i, j):
        if i < 0 or j < 0:
            return 0
        else:
            return lcs_val[(i, j)]

    for i in range(a_len):
        for j in range(b_len):

            if text_a[i] == text_b[j]:
                opt1 = get_arr_val(i-1, j-1) + 1
            else:
                opt1 = -1

            opt2 = get_arr_val(i-1, j)
            opt3 = get_arr_val(i, j-1)
            opt_list = [opt1, opt2, opt3]
            idx = get_max_idx(opt_list)
            opt = idx + 1
            lcs_type[i, j] = opt
            lcs_val[i, j] = opt_list[idx]

    i = a_len - 1
    j = b_len - 1
    match_log = []
    while i >= 0 and j >= 0:
        opt = lcs_type[i, j]
        if opt == 1:
            match_log.append((i, j))
            i, j = i-1, j-1
        elif opt == 2:
            i, j = i-1, j
        elif opt == 3:
            i, j = i, j-1
        else:
            assert False

    prev_j = -1
    return lcs_val[a_len-1, b_len-1], match_log[::-1]


def split_indexed(text):
    st = 0
    st_list = []
    ed_list = []
    tokens = []
    is_in = False
    def pop(idx):
        ed = idx
        token = text[st:ed]
        st_list.append(st)
        ed_list.append(ed)
        tokens.append(token)

    for idx, c in enumerate(text):
        if is_in:
            if c == " ":
                pop(idx)
                is_in = False
            else:
                pass
        else:
            if c == " ":
                pass
            else:
                is_in = True
                st = idx

    if is_in:
        pop(len(text))
    return tokens, list(zip(st_list, ed_list))






def lcs_runner():
    text1 = 'Ackley, Iowa. Ackley is a city in Franklin and Hardin Counties in the U.S. state of Iowa. The population was 1,589 at the 2010 census.'
    text2 = " Total 2.48 sq mi (6.42 km 2)• Land 2.45 sq mi (6.35 km 2)• Water 0.03 sq mi (0.08 km 2) 1.21%Elevation [4] 1,093 ft (333 m)Population ( 2010) [5]• Total 1,589• Estimate (2016) [6] 1,546• Density 640/sq mi (250/km 2)Time zone CST ( UTC-6)• Summer ( DST) CDT ( UTC-5)ZIP code 50601Area code (s) 641FIPS code 19-00190GNIS feature ID 0454084Website Ackley, Iowa [7]Ackley is a city in Franklin and Hardin Counties in the U. S. state of Iowa. The population was 1,589 at the 2010 census. Contents [ hide ]1 History2 Geography3 Demographics3.1 2010 census3.2 2000 census4 Notable people5 References6 External links History [ edit]This section needs additional citations for verification. Please help improve this article by adding citations to reliable sources. Unsourced material may be challenged and removed. (May 2007) ( Learn how and when to remove this template message)In 1852, immigrants began purchasing farms and settling in the north Hardin County area. In the fall of 1852, L. H. Artedge, a frontiers-man from Indiana staked a claim just north of the Hardin County line and built a cabin close to where Highway 57 now passes. Another settler, Thomas Downs, became the first permanent resident of Ackley. Later his widow sold a strip of land from Butler Street to the four county corner for $3.00 an acre to Minor Gallop. Gallop built a house, just east of Highway 57 which became an inn, a stopover for stagecoaches, and the first post office. Many caravans arrived in anticipation of settling in this area. "

    tokens1, indices1 = split_indexed(text1)
    tokens2, indices2 = split_indexed(text2)
    n, log = lcs(tokens1, tokens2, True)
    print(n)
    print(log)

    for a_idx, b_idx in log:
        st, ed = indices1[a_idx]
        print(text1[st:ed], end = " ")
    print()



def time_measure(text1, text2):
    tokens1 = text1.split()
    tokens2 = text2.split()

    def fn1():
        n, log = lcs(text1, text2, True)

    def fn2():
        n, log = lcs(tokens1, tokens2, True)

    for name, fn in [('text', fn1), ('token', fn2)]:
        st = time.time()
        for i in range(10):
            fn()
        ed = time.time()
        print(name, ed - st)


def main():
    lcs_runner()


if __name__ == "__main__":
    main()