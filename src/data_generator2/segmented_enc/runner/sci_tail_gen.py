from data_generator2.segmented_enc.runner.snli_gen import gen_concat_two_seg
from dataset_specific.mnli.sci_tail import SciTailReaderTFDS


def main():
    reader = SciTailReaderTFDS()
    data_name = "sci_tail_sg1"
    for split in ["validation", "train", "test"]:
        gen_concat_two_seg(reader, data_name, split)



if __name__ == "__main__":
    main()
