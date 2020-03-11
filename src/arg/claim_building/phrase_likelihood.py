from arg.claim_building.count_ngram import load_n_gram_from_pickle, is_single_char_n_gram


def analyze():
    topic = "abortion"
    count_d = {}
    ctf_d = {}

    for n in range(1, 4):
        count_d[n] = load_n_gram_from_pickle(topic, n)
        ctf_d[n] = sum(count_d[n].values())

    n = 2

    bigram = count_d[n]
    skip_count = 0
    for n_gram, cnt in bigram.most_common(1000):
        if is_single_char_n_gram(n_gram):
            skip_count += 1
        else:
            # TODO check if P(w_i|w_{1:i-1}) > L * P(w_i)

            n_gram_prefix = n_gram[:-1]
            n_gram_post_fix = n_gram[-1:]
            n_gram_prob = cnt / ctf_d[n]
            prefix_count = count_d[n-1][n_gram_prefix]
            postfix_count = count_d[1][n_gram_post_fix]
            try:
                p_w_i_bar_1_to_i = cnt / prefix_count
                p_w_i = postfix_count / ctf_d[1]
                factor = p_w_i_bar_1_to_i / p_w_i
                if factor < 10:
                    print("We reject n-gram and take n-1 gram. factor={}".format(factor))
                    print(n_gram, cnt)
                    print(n_gram_prefix)
                    print("P(w_i|w_{1:i-1})", p_w_i_bar_1_to_i)
                    print('P(w_i)', p_w_i)
                    print("")

            except ZeroDivisionError as e:
                #print("prefix_count:", prefix_count)
                pass


if __name__ == "__main__":
    analyze()
