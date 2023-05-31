from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_mmp_split_w_deep_scores


def main():
    has_deep_score = get_mmp_split_w_deep_scores()
    print(" ".join(map(str, has_deep_score)))


if __name__ == "__main__":
    main()