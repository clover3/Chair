from collections import Counter

from trainer_v2.per_project.transparency.mmp.bias.exp1.run_inference_w_keyword_swap import load_car_bias_exp_resource


def main():
    passages, query_list, term_list_set, term_list = load_car_bias_exp_resource()


    df = Counter()
    for p in passages:
        text = p.lower()
        for term in term_list:
            if term in text:
                df[term] += 1

    for k, v in df.most_common():
        print("{}\t{}".format(k, v))


if __name__ == "__main__":
    main()