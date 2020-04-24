from cord.csv_to_galago_indexable import read_csv_and_write
from cord.path_info import meta_data_path


def main():
    meta_data_trec_style_path = '/mnt/nfs/work3/youngwookim/data/cord-19/metadata_trecstyle.xml'
    read_csv_and_write(meta_data_path, meta_data_trec_style_path )


if __name__ == "__main__":
    main()
