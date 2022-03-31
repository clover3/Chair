import logging

from transformers import TFMPNetModel


def main():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    tf_model = TFMPNetModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2", from_pt=True)
    save_path = "C:\\work\\Code\\Chair\\output\\model\\runs\\paraphrase-mpnet-base-v2"
    tf_model.save_pretrained(save_path)


if __name__ == "__main__":
    main()