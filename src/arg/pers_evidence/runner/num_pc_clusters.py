from arg.perspectives.load import splits, enum_perspective_clusters_for_split


def main():
    d = {}
    for split in splits:
        pc_clusters = list(enum_perspective_clusters_for_split(split))

        d[split] = len(pc_clusters)
    print(d)


if __name__ == "__main__":
    main()
