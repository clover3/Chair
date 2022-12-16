from data_generator.NLI.enlidef import nli_label_list
from dataset_specific.mnli.snli_reader_tfds import NLIByTFDS


class SciTailReaderTFDS(NLIByTFDS):
    def __init__(self):
        dataset_name = "sci_tail"
        super(SciTailReaderTFDS, self).__init__(dataset_name, nli_label_list)



def main():
    print(SciTailReaderTFDS().get_data_size("train"))
    # 23097

if __name__ == "__main__":
    main()