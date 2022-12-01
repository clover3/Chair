
ists_split_list = ["train", "test"]
ists_genre_list = ["headlines", "images", "answers-students"]


def ists_enum_split_genre_combs():
    for split in ists_split_list[::-1]:
        for genre in ists_genre_list:
            if split == "train" and genre == "answers-students":
                pass
            else:
                yield split, genre