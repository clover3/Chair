import os

from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import get_deep_model_score_path


def main():
    print("Does not exists: ")
    for i in get_valid_mmp_split():
        check_path = get_deep_model_score_path(i)
        if not os.path.exists(check_path):
            print(check_path)




def get_valid_mmp_split():
    yield from range(0, 109)
    yield from range(113, 119)



if __name__ == "__main__":
    main()