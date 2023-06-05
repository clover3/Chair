from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.candidate1.run_candidate1_remain import \
    get_missing_job_info


def main():
    d = get_missing_job_info()
    for key in d:
        print(key)


if __name__ == "__main__":
    main()