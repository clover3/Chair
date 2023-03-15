from trainer_v2.per_project.transparency.splade_regression.modeling.bert_sparse_encoder import DummySparseEncoder


def main():
    vocab_size = 30522
    dataset_info = {
        "max_seq_length": 256,
        "max_vector_indices": 512,
        "vocab_size": vocab_size
    }
    model = DummySparseEncoder(dataset_info)
    model.model.save("output/dummy_sparse_encoder")


if __name__ == "__main__":
    main()